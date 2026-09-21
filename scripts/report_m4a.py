"""M4a 완료 증거 — PPO 학습 곡선 표 + M3/선생님 대비 표 + 목표 판정.

    env -u PYTHONPATH .venv/bin/python scripts/report_m4a.py \
        --run runs/omen/2026-09-21-ppo --m3 runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
        --out docs/reports/m4a-ppo.md --eval-seeds 3

`--skip-eval` 은 정책을 실제로 몰아 보는 절(비교 표·항목별 위반 표·목표 판정)을 건너뛰고
`log.jsonl` 만으로 학습 곡선 표만 채운 뼈대를 만든다(짧은 스모크 실행을 빠르게 확인하는 용도,
테스트가 이걸로 돈다).

M4a 학생 체크포인트(`ac-best.pt`)는 가치 머리가 붙은 `ActorCritic` 이라 `ActorCritic.load(path)`
로 읽고 `.policy` 를 꺼낸다 — `DrivePolicy.load` 로는 못 읽는다(cfg 모양이 다르다). M3 체크포인트는
순수 정책이라 `DrivePolicy.load` 가 맞다.

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json`·`stage2.json` 은 판 이름을 그대로
공유하고 `signals` 만 다르므로, 합쳐 넘기면 `VtdDriveEnv` 의 세계 캐시가 죽는다(`train_ppo.py` 의
같은 주의사항 참고). 판은 `load_curriculum` 으로 단계별로 직접 읽는다.
"""
import argparse
import datetime
import json
import os
import platform
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

# 목표 문턱 — 계획 문서 자체는 "네 줄" 만 요구하고 숫자를 안 주므로, 이번 작업 지시가 확정한
# 값을 여기 상수로 고정한다(스펙 §6.4 완주 기준 + 이번 지시의 항목별 상한).
GOAL_RATE_MIN = 0.9
SCORE_SLACK = 10.0
ITEM2_MAJOR_MAX = 4      # 단계 ①, 항목②(보호구역 속도) 중대
ITEM7_MAJOR_MAX = 19     # 단계 ②, 항목⑦(적색 정지) 중대
STAGE1_LABEL, STAGE2_LABEL = "stage1", "stage2"


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


def _curve_table(rows: list, stage_labels: list) -> list:
    header = ["스텝", "iter"]
    for i, _label in enumerate(stage_labels, start=1):
        header += [f"단계{_circled(i)} 완주율", f"단계{_circled(i)} 점수(완주 0점 기준)"]
    header += ["explained_variance", "log_std", "모방 계수", "approx_kl"]
    lines = ["", "### 학습 곡선(평가 시점마다 — 학습 중 주기 평가는 **시드 1개**)", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for r in rows:
        cells = [str(r["step"]), str(r["iter"])]
        for label in stage_labels:
            st = r["stages"][label]
            cells += [_fmt_pct(st["goal_rate"]), _fmt_score(st["mean_score"])]
        ev = f"{r['explained_variance']:.3f}" if r.get("explained_variance") is not None else "—"
        log_std = ", ".join(f"{x:.3f}" for x in r["log_std"])
        cells += [ev, log_std, f"{r['imitation_coef']:.3f}", f"{r['approx_kl']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
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


def _goal_lines(m4a_ev: dict, teacher_ev: dict) -> list:
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
    item2_ok = item2_major <= ITEM2_MAJOR_MAX
    item7_ok = item7_major <= ITEM7_MAJOR_MAX

    def verdict(ok):
        return "달성" if ok else "미달"

    return ["", "## 목표 판정", "",
            f"- 목표: 완주율 — 단계①② 모두 ≥{_fmt_pct(GOAL_RATE_MIN)} → **{verdict(goal_ok)}** "
            f"(실측: 단계① {_fmt_pct(s1['goal_rate'])}, 단계② {_fmt_pct(s2['goal_rate'])})",
            f"- 목표: 점수 — 완주 못 한 판 0점 기준으로 선생님보다 {SCORE_SLACK:.0f}점 넘게 낮지"
            f" 않을 것 → **{verdict(score_ok)}** (실측: 단계① {_fmt_score(s1['mean_score'])} vs 선생님"
            f" {_fmt_score(t1['mean_score'])}, 단계② {_fmt_score(s2['mean_score'])} vs 선생님"
            f" {_fmt_score(t2['mean_score'])})",
            f"- 목표: 단계② 항목⑦({rs.score_fma.ITEMS[7]}) 중대 위반 ≤{ITEM7_MAJOR_MAX}건 → "
            f"**{verdict(item7_ok)}** (실측 {item7_major}건)",
            f"- 목표: 단계① 항목②({rs.score_fma.ITEMS[2]}) 중대 위반 ≤{ITEM2_MAJOR_MAX}건 → "
            f"**{verdict(item2_ok)}** (실측 {item2_major}건)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train_ppo.py 가 남긴 실행 폴더(log.jsonl, ac-best.pt)")
    ap.add_argument("--m3", default=None, help="비교할 M3 DrivePolicy 체크포인트(policy-rN.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="목표 판정에 쓰는 마지막 전체 평가 시드 수 — 학습 중 주기 평가(log.jsonl, 시드"
                        " 1개)와는 별개로 이 스크립트가 다시 돈다")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="train_ppo.py 실행 때 준 --seed 값 — log.jsonl 에 안 남으므로 알고 있으면"
                        " 직접 넘긴다(성적표에 그대로 적는다, 없으면 '기록 안 됨')")
    ap.add_argument("--skip-eval", action="store_true",
                    help="정책을 실제로 몰아 보는 비교·위반·목표 판정을 건너뛰고 학습 곡선 표만 만든다"
                        "(테스트·빠른 확인용)")
    a = ap.parse_args()
    if not a.skip_eval and not a.m3:
        ap.error("--skip-eval 이 아니면 --m3(비교할 M3 체크포인트)가 필요하다")

    log_path = os.path.join(a.run, "log.jsonl")
    if not os.path.exists(log_path):
        ap.error(f"{log_path} 가 없다")
    with open(log_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        ap.error(f"{log_path} 에 평가 기록이 없다")
    last = rows[-1]
    stage_labels = list(last["stages"])
    curricula = _curricula_from_log(rows)

    lines = ["# M4a 성적표 — PPO", "",
             f"- 날짜 {datetime.date.today().isoformat()} · 성적표 생성 머신 `{platform.node()}` · "
             f"규칙 스택 `{rs.commit()[:7]}`",
             f"- 학습 실행 `{a.run}`(어느 머신에서 돌렸는지는 이 경로로 안다, 예: `runs/omen/...`) · "
             f"총 스텝 {last['step']} · 걸린 시간(마지막 로그 시점 기준) {last['elapsed_s']:.0f}초 · "
             f"반복(iter) {last['iter']}회",
             "- 학습 시드: " + (f"{a.train_seed}(직접 입력값 — log.jsonl 에는 학습 시드가 안 남는다)"
                              if a.train_seed is not None
                              else "기록 안 됨(log.jsonl 에 학습 시드가 없다 — 알고 있으면 --train-seed 로"
                                   " 넘겨라)"),
             "- 학습에 쓴 커리큘럼: " + (", ".join(f"`{os.path.relpath(p, REPO)}`" for _, p in curricula)
                                    or "(log.jsonl 이 비어 있어 알 수 없다)"),
             "- 학습 곡선 표의 평가는 학습 스크립트가 값싸게 자주 도는 주기 평가로 **시드 1개**다. "
             f"아래 목표 판정에 쓰는 수치는 그와 별개로 이 성적표 생성 스크립트가 시드 **{a.eval_seeds}개**"
             "로 다시 돌린 마지막 전체 평가다."]

    lines += _curve_table(rows, stage_labels)

    if a.skip_eval:
        lines += ["", "## 목표 판정", "",
                  "`--skip-eval` 로 만들어 정책을 실제로 몰아 보지 않았다 — 완주율·점수·항목별 위반"
                  " 목표 판정은 보류다(학습 곡선 표만 유효하다).",
                  "", "### 항목별 위반 — 평가 생략", "", "`--skip-eval` 이라 위반 집계가 없다."]
    else:
        dev = pick_device(a.device)
        eval_seeds = tuple(range(a.eval_seeds))
        stage_paths = dict(curricula)
        for label in stage_labels:
            if not os.path.exists(stage_paths[label]):
                ap.error(f"커리큘럼 파일을 못 찾았다: {stage_paths[label]}(라벨 '{label}')")
        boards_by_stage = {label: load_curriculum(stage_paths[label])[1] for label in stage_labels}

        ac = ActorCritic.load(os.path.join(a.run, "ac-best.pt"), device=dev)
        ac.eval()
        m4a_policy = ac.policy
        m3_policy = DrivePolicy.load(a.m3, device=dev)

        m4a_ev = {label: evaluate_policy(m4a_policy, boards_by_stage[label], seeds=eval_seeds)
                  for label in stage_labels}
        m3_ev = {label: evaluate_policy(m3_policy, boards_by_stage[label], seeds=eval_seeds)
                 for label in stage_labels}
        teacher_ev = {label: evaluate_teacher(boards_by_stage[label], seeds=eval_seeds)
                      for label in stage_labels}

        lines += _comparison_lines(stage_labels, m3_ev, m4a_ev, teacher_ev, a.eval_seeds)
        lines += _violation_table_lines(stage_labels, m3_ev, m4a_ev, teacher_ev)
        lines += _goal_lines(m4a_ev, teacher_ev)

    lines += ["", "## M4b 로 미루는 것", "",
              "- 커리큘럼 단계 ③④⑤(사물·정지차 / 보행자·교통 / 연습코스 전체)와 자동 진급 — 액터가"
              " 있는 판을 만들어야 한다(M2a 의 시나리오 도구를 쓴다).",
              "- 두 머신 운용의 나머지(두 머신에서 동시에 다른 설정을 돌리고 결과를 모으기, `nice` 규칙).",
              "- 스텝 예산 계획과 하이퍼파라미터 탐색(엔트로피 계수·클립·롤아웃 길이).",
              "- 같은 시드에서 같은 학습 곡선이 나오는지(학습 반복 재현성) 확인.",
              "", "산출물(로그·체크포인트)은 `runs/` 아래에 있고 레포에는 넣지 않는다."]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"report": a.out, "rows": len(rows), "skip_eval": a.skip_eval}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
