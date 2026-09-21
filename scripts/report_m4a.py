"""M4a 완료 증거 — PPO 학습 곡선 표 + M3/선생님 대비 표 + 목표 판정 + 여러 실행 비교.

    env -u PYTHONPATH .venv/bin/python scripts/report_m4a.py \
        --run runs/omen/2026-09-21-ppo --compare runs/omen/2026-09-22-ppo-entropy \
        --m3 runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
        --out docs/reports/m4a-ppo.md --eval-seeds 3 --notes docs/reports/m4a-ppo-notes.md

`--skip-eval` 은 정책을 실제로 몰아 보는 절(비교 표·항목별 위반 표·목표 판정)을 건너뛰고
`log.jsonl` 만으로 학습 곡선 표·실행 비교 표만 채운 뼈대를 만든다(짧은 스모크 실행을 빠르게
확인하는 용도, 테스트가 이걸로 돈다).

M4a 학생 체크포인트(`ac-best.pt`)는 가치 머리가 붙은 `ActorCritic` 이라 `ActorCritic.load(path)`
로 읽고 `.policy` 를 꺼낸다 — `DrivePolicy.load` 로는 못 읽는다(cfg 모양이 다르다). M3 체크포인트는
순수 정책이라 `DrivePolicy.load` 가 맞다.

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json`·`stage2.json` 은 판 이름을 그대로
공유하고 `signals` 만 다르므로, 합쳐 넘기면 `VtdDriveEnv` 의 세계 캐시가 죽는다(`train_ppo.py` 의
같은 주의사항 참고). 판은 `load_curriculum` 으로 단계별로 직접 읽는다.

`--compare` 로 준 다른 실행들은 요약 비교 표(실행마다 한 행)와 그 실행 자신의 학습 곡선 표에만
나온다 — M3/선생님 대비 표·항목별 위반 표·목표 네 줄 판정은 `--run`(이번에 채점하는 실행) 하나만
싣는다(그게 "M4a 가 끝났는가"를 묻는 절이라 여러 실행을 합쳐 물을 수 없다).
"""
import argparse
import dataclasses
import datetime
import glob
import json
import os
import platform
import re
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.rl.ppo import PPOConfig  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

# 목표 문턱 — 계획 문서 자체는 "네 줄" 만 요구하고 숫자를 안 주므로, 이번 작업 지시가 확정한
# 값을 여기 상수로 고정한다(스펙 §6.4 완주 기준 + 이번 지시의 항목별 상한).
GOAL_RATE_MIN = 0.9
SCORE_SLACK = 10.0
ITEM2_MAJOR_MAX = 4      # 단계 ①, 항목②(보호구역 속도) 중대
ITEM7_MAJOR_MAX = 19     # 단계 ②, 항목⑦(적색 정지) 중대
STAGE1_LABEL, STAGE2_LABEL = "stage1", "stage2"

# `train_ppo.py` 의 `hparams`(=`vars(argparse 결과)`) 에는 `--envs`·`--seed`·`--out` 처럼 실행마다
# 자연히 다른 값도 섞여 있다. `--envs` 는 기본값 자체가 `os.cpu_count() - 2` 라 이 스크립트를 돌리는
# 머신에서 다시 계산하면 실행 당시(OMEN)와 달라 보여 "바뀐 값"으로 잘못 잡힌다. 그래서 비교는
# `PPOConfig` 필드와 이름이 같은, 실제로 CLI 로 연 네 하이퍼파라미터로만 좁힌다(train_ppo.py 의
# `--entropy-coef`/`--lr`/`--target-kl`/`--imitation-half-life` 주석이 "이 네 개" 라 부르는 바로 그것).
CLI_HPARAM_KEYS = ("lr", "entropy_coef", "target_kl", "imitation_half_life")


def _circled(n: int) -> str:
    return chr(0x2460 + n - 1)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _fmt_score(x: float) -> str:
    return f"{x:.1f}"


def _distinct_episodes(ev: dict) -> int:
    """서로 다른(코스, 결과, 걸음수) 조합 수 — `run_dagger.py._distinct_episodes` 와 같은 정의

    (정책이 결정적이면 액터·신호가 고정인 판은 시드를 바꿔도 같은 판이 나온다).
    """
    return len({(e.board, e.outcome, e.steps) for e in ev["episodes"]})


def _read_rows(run_dir: str) -> list:
    """`<run_dir>/log.jsonl` 을 읽는다 — 없으면(학습이 아직 시작 전이거나 첫 평가 전이면) 빈 리스트.

    `--compare` 로 진행 중인 실행을 넘길 수 있어서(2026-09-22 실측 — 두 번째 OMEN 실행이 도는 동안
    성적표를 다시 만들어야 했다) 파일이 없는 것 자체는 에러가 아니다. `--run`(채점 대상)만 main() 에서
    따로 필수로 확인한다.
    """
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _curricula_from_log(rows: list) -> list:
    """`log.jsonl` 의 `stages` 키에서 커리큘럼 파일 경로를 되찾는다.

    `train_ppo.py` 는 `--curricula` 인자 자체를 파일로 남기지 않는다 — `_stage_boards()` 가
    라벨(요약 딕셔너리의 키)을 커리큘럼 파일 이름(확장자 없이)으로 쓰는 규칙을 거꾸로 쓴다.
    기본 커리큘럼(`stage1.json`, `stage2.json`) 밖의 이름을 쓴 실행이면 그 파일이 실제로
    `curricula/` 아래 그 이름으로 있어야 한다.
    """
    if not rows:
        return []
    return [(label, os.path.join(REPO, "curricula", f"{label}.json")) for label in rows[-1]["stages"]]


def _hparam_diff(hparams) -> str:
    """`CLI_HPARAM_KEYS` 만 `PPOConfig()` 기본값과 비교해 바뀐 것만 "key=value" 로 나열한다."""
    if not hparams:
        return "기록 안 됨(이 실행은 hparams 로깅 이전 버전으로 돌았다)"
    defaults = dataclasses.asdict(PPOConfig())
    diffs = [f"{k}={hparams[k]}" for k in CLI_HPARAM_KEYS if k in hparams and hparams[k] != defaults[k]]
    return ", ".join(diffs) if diffs else "(기본값)"


def _detect_best_step(run_dir: str):
    """`ac-best.pt` 가 `ac-<step>.pt` 중 어느 것과 같은 체크포인트인지 state_dict 를 비교해 찾는다.

    학습이 끝난 뒤 뽑은 `best_step` 은 표준출력에 찍는 요약 JSON 에만 있고 파일 어디에도 안 남는다
    (`train_ppo.py` 의 `summary` 는 stdout 전용). 그 숫자를 손으로 옮겨 적지 않으려면 체크포인트
    자체를 비교해 되찾아야 한다 — CPU 로만 읽는다(환경 시뮬레이션이 없어 device 를 안 가린다).
    """
    best_path = os.path.join(run_dir, "ac-best.pt")
    if not os.path.exists(best_path):
        return None
    best_state = ActorCritic.load(best_path, device="cpu").state_dict()
    for path in sorted(glob.glob(os.path.join(run_dir, "ac-*.pt"))):
        m = re.fullmatch(r"ac-(\d+)\.pt", os.path.basename(path))
        if not m:
            continue
        cand_state = ActorCritic.load(path, device="cpu").state_dict()
        if set(cand_state) == set(best_state) and all(
                torch.equal(cand_state[k], best_state[k]) for k in best_state):
            return int(m.group(1))
    return None


def _train_seed_line(hparams, train_seed_arg) -> str:
    """학습 시드 — 최우선 출처는 `hparams["seed"]`(`train_ppo.py` 가 실제로 `torch.manual_seed`

    에 준 값)다. 이게 있는 실행(hparams 로깅 이후)에서 "기록 안 됨"이라 말하는 건 산출물에
    이미 있는 값을 없다고 말하고 사람에게 `--train-seed` 로 다시 손으로 넣으라고 시키는
    것과 같다(2026-09-22 리뷰) — README 의 "숫자를 손으로 옮겨 적지 않는다" 원칙에 정면으로
    어긋난다. `hparams` 자체가 없는 실행(로깅 이전, 예: `runs/omen/2026-09-21-ppo`)에서만
    `--train-seed` 폴백을 쓴다. 어느 경로로 얻었는지 항상 괄호로 밝힌다.
    """
    if hparams and hparams.get("seed") is not None:
        return f"{hparams['seed']}(log.jsonl 의 hparams 에서 읽음)"
    if train_seed_arg is not None:
        return (f"{train_seed_arg}(직접 입력값 — 이 실행은 hparams 로깅 이전이라 log.jsonl 에 "
                "시드가 없다)")
    return "기록 안 됨(log.jsonl 에 hparams 가 없고 --train-seed 도 안 줬다)"


def _best_step_caveat(rows: list, best_step) -> str:
    """`best_step` 이 전체 스텝의 절반 미만이면, 그 시점의 모방 계수와 함께 사실을 그대로 적는다.

    이건 해석이 아니라 사실이다 — 이게 없으면 완주율·점수 줄의 "달성"이 PPO 자체의 성과처럼
    읽힌다(2026-09-22 지시: 507k/3M, 모방 계수 0.839 인 체크포인트가 뽑혔다).
    """
    if best_step is None or not rows:
        return ""
    total_steps = rows[-1]["step"]
    if total_steps <= 0 or best_step >= total_steps / 2:
        return ""
    match = next((r for r in rows if r.get("step") == best_step), None)
    coef = f"{match['imitation_coef']:.3f}" if match and "imitation_coef" in match else "알 수 없음"
    pct = best_step / total_steps * 100
    return (f"선정된 체크포인트가 {best_step} 스텝(전체 {total_steps} 의 {pct:.0f}%) 지점이고, 그때"
            f" 모방 계수가 {coef} 였다 — 위 달성이 PPO 자체의 성과라기보다 M3 워밍스타트에 가까울"
            " 수 있다는 뜻이다.")


def _curve_table(rows: list, stage_labels: list) -> list:
    """평균 보상(`mean_reward`) 도 완주율·점수와 나란히 단계마다 한 열 — 계획서가 명시한 열이고,

    관찰 노트("단계① 평균 보상 102.5 → 22.5")가 바로 이 숫자를 인용하므로 성적표에서도 그
    숫자를 확인할 수 있어야 한다(2026-09-22 리뷰). `mean_reward` 는 `train_ppo.py` 의
    `PUBLIC_KEYS` 에 있어 `log.jsonl` 각 줄의 `stages.<라벨>` 에 이미 들어 있다.
    """
    header = ["스텝", "iter"]
    for i, _label in enumerate(stage_labels, start=1):
        header += [f"단계{_circled(i)} 완주율", f"단계{_circled(i)} 점수(완주 0점 기준)",
                   f"단계{_circled(i)} 평균 보상"]
    header += ["explained_variance", "log_std", "모방 계수", "approx_kl"]
    lines = ["", "| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for r in rows:
        cells = [str(r["step"]), str(r["iter"])]
        for label in stage_labels:
            st = r["stages"][label]
            cells += [_fmt_pct(st["goal_rate"]), _fmt_score(st["mean_score"]), _fmt_score(st["mean_reward"])]
        ev = f"{r['explained_variance']:.3f}" if r.get("explained_variance") is not None else "—"
        log_std = ", ".join(f"{x:.3f}" for x in r["log_std"])
        cells += [ev, log_std, f"{r['imitation_coef']:.3f}", f"{r['approx_kl']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _summary_table(run_dirs: list, rows_by_run: dict, best_steps: dict, extra_evs: dict,
                   skip_eval: bool) -> list:
    """실행마다 한 행 — 학습 중 주기 평가(시드 1개, log.jsonl)를 기본으로 쓰고, `extra_evs` 에

    그 실행의 라이브 평가(`--skip-eval` 이 아닐 때만 채워진다)가 있으면 그 값으로 덮는다. 항목⑦·②
    중대 건수는 구간-슬롯(sheet) 정보가 필요해 라이브 평가 없이는 절대 못 낸다 — 그때는 "—".
    """
    header = ["실행", "바뀐 하이퍼파라미터", "best_step", "단계① 완주율", "단계① 점수",
              "단계② 완주율", "단계② 점수", "항목⑦ 중대", "항목② 중대", "log_std", "EV"]
    lines = ["", "## 실행 비교", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for rd in run_dirs:
        rows = rows_by_run.get(rd, [])
        name = os.path.basename(os.path.normpath(rd))
        best_step = best_steps.get(rd)
        last = rows[-1] if rows else None
        hp = _hparam_diff(last.get("hparams")) if last else "로그 없음(학습 진행 중이거나 시작 전)"
        stage_vals = dict(last["stages"]) if last else {}
        item7 = item2 = "—"
        live = extra_evs.get(rd)
        if live:
            stage_vals.update(live)
            if STAGE2_LABEL in live:
                item7 = violation_counts(live[STAGE2_LABEL]).get(7, {"major": 0})["major"]
            if STAGE1_LABEL in live:
                item2 = violation_counts(live[STAGE1_LABEL]).get(2, {"major": 0})["major"]
        s1, s2 = stage_vals.get(STAGE1_LABEL), stage_vals.get(STAGE2_LABEL)
        log_std = ", ".join(f"{x:.3f}" for x in last["log_std"]) if last else "—"
        ev_val = (f"{last['explained_variance']:.3f}"
                  if last and last.get("explained_variance") is not None else "—")
        row = [name, hp, str(best_step) if best_step is not None else "미완료(ac-best.pt 없음)",
               _fmt_pct(s1["goal_rate"]) if s1 else "—", _fmt_score(s1["mean_score"]) if s1 else "—",
               _fmt_pct(s2["goal_rate"]) if s2 else "—", _fmt_score(s2["mean_score"]) if s2 else "—",
               str(item7), str(item2), log_std, ev_val]
        lines.append("| " + " | ".join(row) + " |")
    if skip_eval:
        lines += ["", "(`--skip-eval` 이라 항목⑦·② 열은 log.jsonl 만으로 못 채워 `—` 다.)"]
    return lines


def _comparison_lines(stage_labels: list, m3_ev: dict, m4a_ev: dict, teacher_ev: dict, eval_seeds: int) -> list:
    lines = ["", "## 비교 — M3 학생 · M4a 학생 · 선생님", "",
             "| 단계 | 지표 | M3 학생 | M4a 학생 | 선생님 |", "|---|---|---:|---:|---:|"]
    for i, label in enumerate(stage_labels, start=1):
        tag = f"단계{_circled(i)}"
        lines.append(f"| {tag} | 완주율 | {_fmt_pct(m3_ev[label]['goal_rate'])} | "
                     f"{_fmt_pct(m4a_ev[label]['goal_rate'])} | {_fmt_pct(teacher_ev[label]['goal_rate'])} |")
        lines.append(f"| {tag} | 점수(완주 0점 기준) | {_fmt_score(m3_ev[label]['mean_score'])} | "
                     f"{_fmt_score(m4a_ev[label]['mean_score'])} | {_fmt_score(teacher_ev[label]['mean_score'])} |")

    lines += ["", f"### 서로 다른 판 수(시드 {eval_seeds}개 중 실제로 달랐던 것)", ""]
    for i, label in enumerate(stage_labels, start=1):
        n = len(m4a_ev[label]["episodes"])
        note = ("(액터·신호가 고정이라 시드가 달라도 같은 판이 되기 쉽다)" if label == STAGE1_LABEL
                else "(신호 주기 위상이 시드마다 달라 실제로 다른 판이 나올 수 있다)")
        lines.append(f"- 단계{_circled(i)}: M3 {_distinct_episodes(m3_ev[label])}/{n}, "
                     f"M4a {_distinct_episodes(m4a_ev[label])}/{n}, "
                     f"선생님 {_distinct_episodes(teacher_ev[label])}/{n} {note}")
    return lines


def _violation_table_lines(stage_labels: list, m3_ev: dict, m4a_ev: dict, teacher_ev: dict) -> list:
    lines = []
    for i, label in enumerate(stage_labels, start=1):
        sc_m3 = violation_counts(m3_ev[label])
        sc_m4a = violation_counts(m4a_ev[label])
        sc_t = violation_counts(teacher_ev[label])
        items = sorted(set(sc_m3) | set(sc_m4a) | set(sc_t))
        lines += ["", f"### 단계{_circled(i)} — 항목별 위반(구간-슬롯 수)", ""]
        if not items:
            lines.append("M3·M4a·선생님 모두 위반 없음.")
            continue
        lines += ["| 항목 | M3 minor | M3 major | M4a minor | M4a major | 선생님 minor | 선생님 major |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in items:
            a3, a4, at = sc_m3.get(item, {}), sc_m4a.get(item, {}), sc_t.get(item, {})
            name = rs.score_fma.ITEMS.get(item, "?")
            lines.append(f"| {item}({name}) | {a3.get('minor', 0)} | {a3.get('major', 0)} | "
                        f"{a4.get('minor', 0)} | {a4.get('major', 0)} | "
                        f"{at.get('minor', 0)} | {at.get('major', 0)} |")
    return lines


def _item_goal_line(tag: str, item_name: str, major_count: int, max_count: int, eval_seeds: int) -> tuple:
    """`ITEM7_MAJOR_MAX`/`ITEM2_MAJOR_MAX` 는 **시드 3개 합계**를 전제로 만든 절대 건수다.

    단계①은 액터·신호가 고정이라 시드를 바꿔도 같은 판 6개라서, 항목② 건수가 시드 수에
    정확히 비례한다(시드 1개 → 5건, 시드 3개 → 15건 — 2026-09-22 리뷰가 실측으로 잡았다).
    `--eval-seeds` 를 3 이 아닌 값으로 돌리면 판정이 조용히 뒤집힐 수 있으므로, 문턱을 임의로
    스케일하지 않고(원래 기준이 "시드 3개 합계" 라는 사실 자체가 드러나야 한다) 판정을
    보류한다. 시드 3개일 때도 어떤 시드 수의 합계인지는 항상 같이 찍는다.
    """
    if eval_seeds != 3:
        return ("보류", f"- 목표: {tag}({item_name}) 중대 위반 ≤{max_count}건(시드 3개 합계 기준) → "
                        f"**보류** — 이번 평가는 시드 {eval_seeds}개라 문턱(시드 3개 합계 전제)과 "
                        f"단위가 안 맞는다(실측 {major_count}건, 시드 {eval_seeds}개 합계).")
    ok = major_count <= max_count
    verdict = "달성" if ok else "미달"
    return (verdict, f"- 목표: {tag}({item_name}) 중대 위반 ≤{max_count}건 → **{verdict}** "
                     f"(실측 {major_count}건, 시드 {eval_seeds}개 합계)")


def _goal_lines(m4a_ev: dict, teacher_ev: dict, rows: list, best_step, eval_seeds: int) -> list:
    """목표 네 줄(완주율·점수·항목⑦·항목②) — 이 절이 쓰는 유일한 평가는 마지막 전체 평가

    (시드 여러 개, `--eval-seeds`)다. 학습 곡선 표의 주기 평가(시드 1개)는 여기 안 쓴다.
    """
    s1, s2 = m4a_ev.get(STAGE1_LABEL), m4a_ev.get(STAGE2_LABEL)
    t1, t2 = teacher_ev.get(STAGE1_LABEL), teacher_ev.get(STAGE2_LABEL)
    missing = s1 is None or s2 is None or t1 is None or t2 is None
    if missing:
        return ["", "## 목표 판정", "",
                f"단계 라벨이 `{STAGE1_LABEL}`/`{STAGE2_LABEL}` 이 아니라 목표 네 줄을 판정할 수 없다"
                " (log.jsonl 의 stages 키를 확인해라)."]

    item2_major = violation_counts(s1).get(2, {"major": 0})["major"]
    item7_major = violation_counts(s2).get(7, {"major": 0})["major"]
    goal_ok = s1["goal_rate"] >= GOAL_RATE_MIN and s2["goal_rate"] >= GOAL_RATE_MIN
    score_ok = (s1["mean_score"] >= t1["mean_score"] - SCORE_SLACK
                and s2["mean_score"] >= t2["mean_score"] - SCORE_SLACK)
    item7_status, item7_line = _item_goal_line("단계② 항목⑦", rs.score_fma.ITEMS[7], item7_major,
                                               ITEM7_MAJOR_MAX, eval_seeds)
    item2_status, item2_line = _item_goal_line("단계① 항목②", rs.score_fma.ITEMS[2], item2_major,
                                               ITEM2_MAJOR_MAX, eval_seeds)
    achieved = sum([goal_ok, score_ok, item7_status == "달성", item2_status == "달성"])
    held = sum([item7_status == "보류", item2_status == "보류"])

    def verdict(ok):
        return "달성" if ok else "미달"

    head = (f"- 목표 4 줄 중 **{achieved} 줄 달성**"
            + (f"({held} 줄 보류 — 시드 수가 3 이 아니라 항목 문턱과 단위가 안 맞는다)."
               if held else "."))
    lines = ["", "## 목표 판정", "", head, "",
             f"- 목표: 완주율 — 단계①② 모두 ≥{_fmt_pct(GOAL_RATE_MIN)} → **{verdict(goal_ok)}** "
             f"(실측: 단계① {_fmt_pct(s1['goal_rate'])}, 단계② {_fmt_pct(s2['goal_rate'])})",
             f"- 목표: 점수 — 완주 못 한 판 0점 기준으로 선생님보다 {SCORE_SLACK:.0f}점 넘게 낮지"
             f" 않을 것 → **{verdict(score_ok)}** (실측: 단계① {_fmt_score(s1['mean_score'])} vs 선생님"
             f" {_fmt_score(t1['mean_score'])}, 단계② {_fmt_score(s2['mean_score'])} vs 선생님"
             f" {_fmt_score(t2['mean_score'])})"]
    if goal_ok or score_ok:
        caveat = _best_step_caveat(rows, best_step)
        if caveat:
            lines.append(f"  - ⚠️ {caveat}")
    lines += [item7_line, item2_line]
    return lines


def _notes_lines(notes_path) -> list:
    """"관찰과 해석" 절 — 표의 숫자는 전부 이 스크립트가 생성하지만, 무엇이 일어났고 왜인지에

    대한 해석은 생성할 수 없다(2026-09-22 지시). `--notes` 파일 내용을 그대로 삽입만 한다 —
    검증도 가공도 안 한다. 안 주면 정확히 이 한 줄만 남긴다(자리표시자를 알아서 지어내지 않는다).
    """
    lines = ["", "## 관찰과 해석", ""]
    if not notes_path:
        return lines + ["관찰과 해석 — `--notes` 로 주지 않았다"]
    with open(notes_path, encoding="utf-8") as f:
        content = f.read().rstrip("\n")
    return lines + [content]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train_ppo.py 가 남긴 실행 폴더(log.jsonl, ac-best.pt)")
    ap.add_argument("--compare", action="append", default=[],
                    help="비교할 다른 실행 폴더(반복 가능) — 요약 비교 표와 자기 학습 곡선 표에만 나온다"
                        "(M3/선생님 대비·항목별 위반·목표 판정은 --run 하나만 싣는다)")
    ap.add_argument("--notes", default=None,
                    help="'관찰과 해석' 절에 그대로 삽입할 마크다운 파일(가공 없이 그대로 붙인다)")
    ap.add_argument("--m3", default=None, help="비교할 M3 DrivePolicy 체크포인트(policy-rN.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="목표 판정에 쓰는 마지막 전체 평가 시드 수 — 학습 중 주기 평가(log.jsonl, 시드"
                        " 1개)와는 별개로 이 스크립트가 다시 돈다")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="hparams 로깅 이전 실행(log.jsonl 에 hparams.seed 가 없는 실행)일 때만 쓰는"
                        " 폴백 — hparams 가 있으면 거기 적힌 seed 를 그대로 쓰고 이 값은 무시한다")
    ap.add_argument("--skip-eval", action="store_true",
                    help="정책을 실제로 몰아 보는 비교·위반·목표 판정을 건너뛰고 학습 곡선·실행 비교"
                        " 표만 만든다(테스트·빠른 확인용)")
    a = ap.parse_args()
    if not a.skip_eval and not a.m3:
        ap.error("--skip-eval 이 아니면 --m3(비교할 M3 체크포인트)가 필요하다")

    primary_rows = _read_rows(a.run)
    if not primary_rows:
        ap.error(f"{os.path.join(a.run, 'log.jsonl')} 가 없거나 비어 있다")
    last = primary_rows[-1]
    stage_labels = list(last["stages"])
    curricula = _curricula_from_log(primary_rows)

    run_dirs = [a.run] + list(a.compare)
    rows_by_run = {rd: (primary_rows if rd == a.run else _read_rows(rd)) for rd in run_dirs}
    best_steps = {rd: _detect_best_step(rd) for rd in run_dirs}

    # 라이브 평가(비교 표·항목별 위반·목표 판정·실행 비교 표의 항목⑦·② 열이 쓴다) — `--run` 은
    # 한 번만 평가해 두 곳(깊은 비교 절 + 실행 비교 표)에 그대로 재사용한다(중복 평가로 몇 분씩
    # 낭비하지 않는다). `--compare` 실행은 체크포인트가 있을 때만(학습이 끝났을 때만) 평가한다.
    m4a_ev = m3_ev = teacher_ev = None
    extra_evs = {}
    dev = None
    if not a.skip_eval:
        dev = pick_device(a.device)
        eval_seeds = tuple(range(a.eval_seeds))
        stage_paths = dict(curricula)
        for label in stage_labels:
            if not os.path.exists(stage_paths[label]):
                ap.error(f"커리큘럼 파일을 못 찾았다: {stage_paths[label]}(라벨 '{label}')")
        boards_by_stage = {label: load_curriculum(stage_paths[label])[1] for label in stage_labels}

        ac = ActorCritic.load(os.path.join(a.run, "ac-best.pt"), device=dev)
        ac.eval()
        m3_policy = DrivePolicy.load(a.m3, device=dev)

        m4a_ev = {label: evaluate_policy(ac.policy, boards_by_stage[label], seeds=eval_seeds)
                  for label in stage_labels}
        m3_ev = {label: evaluate_policy(m3_policy, boards_by_stage[label], seeds=eval_seeds)
                 for label in stage_labels}
        teacher_ev = {label: evaluate_teacher(boards_by_stage[label], seeds=eval_seeds)
                      for label in stage_labels}
        extra_evs[a.run] = m4a_ev

        for rd in a.compare:
            rows = rows_by_run[rd]
            if not rows or best_steps.get(rd) is None:
                continue   # 학습이 아직 안 끝났다(ac-best.pt 없음) — 요약 표에서 "미완료" 로 남는다
            cand = ActorCritic.load(os.path.join(rd, "ac-best.pt"), device=dev)
            cand.eval()
            evs = {}
            for label, path in _curricula_from_log(rows):
                if os.path.exists(path):
                    _n, boards = load_curriculum(path)
                    evs[label] = evaluate_policy(cand.policy, boards, seeds=eval_seeds)
            extra_evs[rd] = evs

    lines = ["# M4a 성적표 — PPO", "",
             f"- 날짜 {datetime.date.today().isoformat()} · 성적표 생성 머신 `{platform.node()}` · "
             f"규칙 스택 `{rs.commit()[:7]}`",
             f"- 채점 대상 실행 `{a.run}`(어느 머신에서 돌렸는지는 이 경로로 안다, 예: `runs/omen/...`)"
             f" · 총 스텝 {last['step']} · 걸린 시간(마지막 로그 시점 기준) {last['elapsed_s']:.0f}초 · "
             f"반복(iter) {last['iter']}회",
             "- 학습 시드: " + _train_seed_line(last.get("hparams"), a.train_seed),
             "- 학습에 쓴 커리큘럼: " + (", ".join(f"`{os.path.relpath(p, REPO)}`" for _, p in curricula)
                                    or "(log.jsonl 이 비어 있어 알 수 없다)"),
             "- 학습 곡선 표의 평가는 학습 스크립트가 값싸게 자주 도는 주기 평가로 **시드 1개**다. "
             f"아래 목표 판정에 쓰는 수치는 그와 별개로 이 성적표 생성 스크립트가 시드 **{a.eval_seeds}개**"
             "로 다시 돌린 마지막 전체 평가다."]

    # 학습 곡선 표 — 실행마다(비교 대상 포함) 나란히
    for rd in run_dirs:
        rows = rows_by_run[rd]
        name = os.path.basename(os.path.normpath(rd))
        lines += ["", f"### 학습 곡선 — 실행 `{name}`(`{rd}`)"]
        if not rows:
            lines += ["", "로그가 없다(학습이 아직 시작 전이거나 첫 평가 전이다)."]
            continue
        lines += _curve_table(rows, list(rows[-1]["stages"]))

    lines += _summary_table(run_dirs, rows_by_run, best_steps, extra_evs, a.skip_eval)

    if a.skip_eval:
        lines += ["", "## 목표 판정", "",
                  "`--skip-eval` 로 만들어 정책을 실제로 몰아 보지 않았다 — 완주율·점수·항목별 위반"
                  " 목표 판정은 보류다(학습 곡선 표·실행 비교 표만 유효하다).",
                  "", "### 항목별 위반 — 평가 생략", "", "`--skip-eval` 이라 위반 집계가 없다."]
    else:
        lines += _comparison_lines(stage_labels, m3_ev, m4a_ev, teacher_ev, a.eval_seeds)
        lines += _violation_table_lines(stage_labels, m3_ev, m4a_ev, teacher_ev)
        lines += _goal_lines(m4a_ev, teacher_ev, primary_rows, best_steps[a.run], a.eval_seeds)

    lines += _notes_lines(a.notes)

    lines += ["", "## M4b 로 미루는 것", "",
              "- 커리큘럼 단계 ③④⑤(사물·정지차 / 보행자·교통 / 연습코스 전체)와 자동 진급 — 액터가"
              " 있는 판을 만들어야 한다(M2a 의 시나리오 도구를 쓴다).",
              "- 두 머신 운용의 나머지(두 머신에서 동시에 다른 설정을 돌리고 결과를 모으기, `nice` 규칙).",
              "- 스텝 예산 계획과 하이퍼파라미터 탐색(엔트로피 계수·클립·롤아웃 길이) — 이 성적표의"
              " 실행 비교 표가 그 탐색의 첫 결과다.",
              "- 같은 시드에서 같은 학습 곡선이 나오는지(학습 반복 재현성) 확인.",
              "", "산출물(로그·체크포인트)은 `runs/` 아래에 있고 레포에는 넣지 않는다."]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"report": a.out, "rows": len(primary_rows), "skip_eval": a.skip_eval,
                      "runs": run_dirs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
