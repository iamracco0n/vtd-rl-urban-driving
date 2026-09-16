"""M3 — DAgger 라운드 반복과 성적표.

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --out runs/lab-main/$(date +%F)-dagger --rounds 5 --seeds 2 --workers 12
    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --smoke --out /tmp/smoke

성적표만 다시 만들 때(라운드를 다시 안 돌림):

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --report-only \
        --out runs/lab-main/2026-09-17-dagger-fix2 --history ... --history-note "..."
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import platform
import sys
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.collect import collect_episode  # noqa: E402
from vtd_rl.policy.dataset import load_dir, load_shard, save_shard  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher  # noqa: E402
from vtd_rl.policy.net import DrivePolicy, PolicyConfig  # noqa: E402
from vtd_rl.policy.train import TrainConfig, train_epochs  # noqa: E402
from vtd_rl.world.board import load_board, load_curriculum, slice_board  # noqa: E402

BETAS = [1.0, 0.5, 0.25, 0.1, 0.0]
H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STOPPED_SPEED = 0.02   # vec[:,0] = ego 속도/25 클립값 — 0.02 는 약 0.5 m/s 이하, "거의 정지"


def _collect_job(job):
    curriculum, name, seed, beta, policy_path, out, rnd, smoke = job
    board = _boards(curriculum, smoke, name)[0]
    policy = DrivePolicy.load(policy_path) if policy_path else None
    shard = collect_episode(board, policy=policy, beta=beta, seed=seed)
    save_shard(shard, os.path.join(out, f"r{rnd}-{name}-s{seed}.npz"))
    return shard.meta


def _boards(curriculum, smoke, only=None):
    if smoke:
        boards = [slice_board(load_board(H), 0.0, 250.0, "H_0_250")]
    else:
        _n, boards = load_curriculum(curriculum)
    return [b for b in boards if only is None or b.name == only]


def _new_net(smoke, dev):
    cfg = PolicyConfig(trunk=(64, 64)) if smoke else PolicyConfig()
    return DrivePolicy(cfg).to(dev)


def _stall_start_count(data_dir, rnd, names, num_seeds) -> int:
    """이 라운드에 새로 모은 조각만 다시 읽어(재수집 없이) 센다 —

    자차가 거의 정지(`vec[:,0] < STOPPED_SPEED`)했는데 선생님 라벨은 출발하라고 한
    (`control[:,1] > 0`, 곧 양의 가속) 표본 수. 학생이 멈춰서 못 움직이는 실패 양상을
    교정하는 바로 그 신호이므로, 라운드가 갈수록 느는지 주는지가 M4 의 출발점이다.
    """
    n = 0
    for name in names:
        for s in range(num_seeds):
            seed = 1000 * rnd + s          # _collect_job 이 파일명에 쓰는 것과 같은 시드 계산
            shard = load_shard(os.path.join(data_dir, f"r{rnd}-{name}-s{seed}.npz"))
            n += int(((shard.vec[:, 0] < STOPPED_SPEED) & (shard.control[:, 1] > 0.0)).sum())
    return n


def _distinct_episodes(ev: dict) -> int:
    """서로 다른(코스, 결과, 걸음수) 조합 수 — 정책이 결정적이면 액터·신호가 고정인 판은

    시드를 바꿔도 똑같은 판이 나온다(단계 ①). "완주율 X%" 가 실제로 몇 가지 서로 다른 판을
    본 것인지 밝혀 둔다.
    """
    return len({(e.board, e.outcome, e.steps) for e in ev["episodes"]})


def _teacher_caveat(teacher2: dict) -> str:
    """단계 ② 선생님의 위반을 감점 시트에서 직접 뽑아 낸다(레포에 박아 넣지 않는다).

    Task 1 에서 이미 본 대로 신호 주기에서는 선생님도 무결하지 않다(빨간불 위반·차선 관련 경미 감점) —
    그래도 전부 완주한다. "선생님보다 10점 이내"를 비교할 때 그 선생님 점수 자체가 이미 이런 감점을
    포함한 값임을 밝혀 둔다.
    """
    by_board: dict = {}
    for e in teacher2["episodes"]:
        for section in e.sheet:
            for item, grade in section.items():
                by_board.setdefault(e.board, set()).add((item, grade))
    if not by_board:
        return "선생님은 단계 ②(신호 주기)에서도 감점 없이 전부 완주했다."
    parts = [f"{board} " + ", ".join(f"항목{i}({g})" for i, g in sorted(items))
             for board, items in sorted(by_board.items())]
    return ("선생님도 단계 ②(신호 주기)에서 무결하지는 않다 — " + "; ".join(parts) +
            f" — 그래도 여섯 코스 모두 완주했다(완주율 {teacher2['goal_rate']*100:.0f}%, "
            f"점수 {teacher2['mean_score']:.1f}).")


def _final_outcome_lines(ev1: dict) -> list:
    by_board: dict = {}
    for e in ev1["episodes"]:
        by_board.setdefault(e.board, []).append(f"{e.outcome}({e.steps}보)")
    lines = ["", "### 마지막 라운드 — 단계 ① 코스별 결과(시드 순)", "",
             "| 코스 | 결과(걸음수) |", "|---|---|"]
    for board, outs in by_board.items():
        lines.append(f"| {board} | {' '.join(outs)} |")
    return lines


def _history_lines(history_paths, history_notes) -> list:
    if not history_paths and not history_notes:
        return []
    lines = ["", "## 이전 실행(참고)"]
    for note in history_notes:
        lines += ["", f"- {note}"]
    if history_paths:
        lines += ["", "아래는 이번에 고친 결함(라운드마다 웜스타트로 에폭이 쌓임, `log_std` 붕괴)이 있던 상태에서"
                 " 나온 예전 실행이다. 참고용으로 남겨 둔다 — 지금 성적표의 판정 기준이 아니다."]
    for path in history_paths:
        # 히스토리 파일 하나가 깨졌다고(없음·JSON 오류·행 형식 다름) 지금 막 나온 실행 성적표까지
        # 못 쓰게 되면 안 된다 — 넓게 잡아 그 파일만 건너뛴다.
        try:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            table = ["", f"### `{path}`", "",
                    "| 라운드 | β | 수집 판(완주) | 누적 표본 | 손실 | 단계 ① 완주율 | 단계 ① 점수 |"
                    " 단계 ② 완주율 | 단계 ② 점수 |",
                    "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                table.append(f"| {r['round']} | {r['beta']} | {r['episodes']}({r['collect_goal']}) | "
                             f"{r['samples']} | {r['train']['loss']:.3f} | "
                             f"{r['stage1']['goal_rate']*100:.0f}% | {r['stage1']['mean_score']:.1f} | "
                             f"{r['stage2']['goal_rate']*100:.0f}% | {r['stage2']['mean_score']:.1f} |")
            lines += table
        except Exception as exc:  # noqa: BLE001 — 여기서 다 삼켜서 성적표 생성 자체를 지킨다
            lines += ["", f"### `{path}` — 읽지 못함: {type(exc).__name__}: {exc}"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=os.path.join(REPO, "docs", "reports", "m3-dagger.md"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--history", action="append", default=[],
                    help="이전 실행의 log.jsonl 경로(반복 가능) — 성적표에 '이전 실행' 절로 남긴다")
    ap.add_argument("--history-note", action="append", default=[],
                    help="'이전 실행' 절에 그대로 적을 문장(반복 가능) — 표로 담기 어려운 맥락용")
    ap.add_argument("--report-only", action="store_true",
                    help="라운드를 다시 돌리지 않고 --out 의 기존 log.jsonl·체크포인트만으로 성적표를 다시 만든다")
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.seeds, a.epochs, a.eval_seeds, a.workers = 1, 1, 2, 1, 1
    if a.rounds < 1:
        ap.error("--rounds 는 1 이상이어야 한다")

    data_dir = os.path.join(a.out, "data")
    log_path = os.path.join(a.out, "log.jsonl")
    dev = pick_device(a.device)
    stage1 = os.path.join(REPO, "curricula", "stage1.json")
    stage2 = os.path.join(REPO, "curricula", "stage2.json")
    eval_seeds = tuple(range(a.eval_seeds))

    if a.report_only:
        if not os.path.exists(log_path):
            ap.error(f"--report-only 인데 {log_path} 가 없다")
        with open(log_path, encoding="utf-8") as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        if not rounds:
            ap.error(f"{log_path} 에 라운드가 없다")
    else:
        os.makedirs(data_dir, exist_ok=True)
        rounds, t0 = [], time.perf_counter()

        for rnd in range(a.rounds):
            beta = BETAS[rnd] if rnd < len(BETAS) else 0.0
            policy_path = os.path.join(a.out, f"policy-r{rnd-1}.pt") if rnd else None
            names = [b.name for b in _boards(stage1, a.smoke)]
            jobs = [(stage1, name, 1000 * rnd + s, beta, policy_path, data_dir, rnd, a.smoke)
                    for name in names for s in range(a.seeds)]
            t_collect = time.perf_counter()
            if a.workers > 1:
                with mp.get_context("spawn").Pool(a.workers) as pool:
                    metas = pool.map(_collect_job, jobs)
            else:
                metas = [_collect_job(j) for j in jobs]
            collect_s = time.perf_counter() - t_collect
            stall_start = _stall_start_count(data_dir, rnd, names, a.seeds)

            dataset = load_dir(data_dir)
            # DAgger 는 라운드마다 누적 데이터셋에 예측기를 "다시" 짓는다 — 이전 라운드 그물을 웜스타트로
            # 이어 쓰면 라운드마다 에폭이 쌓여(라운드 7 이면 8 라운드 * 8 에폭 = 64 에폭) 조향 log_std 가
            # 무너지고 가속 머리가 굶는다(수정 라운드 — 측정: 웜스타트 policy-r7 완주율 0%, 새로 지은 그물
            # 8 에폭 완주율 100%, 같은 데이터). 판을 몰 때 쓰는 이전 라운드 체크포인트(policy_path, β 혼합)는
            # 그대로 두고, 학습만 매 라운드 새 그물로 한다.
            net_r = _new_net(a.smoke, dev)
            train = train_epochs(net_r, dataset, TrainConfig(epochs=a.epochs, seed=rnd), device=dev)
            net_r.save(os.path.join(a.out, f"policy-r{rnd}.pt"))

            ev1_r = evaluate_policy(net_r, _boards(stage1, a.smoke), seeds=eval_seeds)
            ev2_r = evaluate_policy(net_r, _boards(stage2, a.smoke), seeds=eval_seeds)
            row = {"round": rnd, "beta": beta, "episodes": len(metas),
                   "collect_goal": sum(1 for m in metas if m["outcome"] == "goal"),
                   "samples": len(dataset), "collect_s": collect_s, "train": train,
                   "stall_start": stall_start,
                   "stage1": {k: ev1_r[k] for k in ("goal_rate", "mean_score", "mean_reward")},
                   "stage2": {k: ev2_r[k] for k in ("goal_rate", "mean_score", "mean_reward")}}
            rounds.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 마지막 라운드 체크포인트를 다시 읽어(학습 도중의 net 객체에 기대지 않고) 상세 평가를 낸다 —
    # --report-only 에서도 그대로 쓸 수 있고, 정상 실행에서도 저장한 체크포인트가 로그의 숫자와
    # 같은 걸 낸다는 확인이 겸사겸사 된다.
    last = rounds[-1]
    net = DrivePolicy.load(os.path.join(a.out, f"policy-r{last['round']}.pt"), device=dev)
    ev1 = evaluate_policy(net, _boards(stage1, a.smoke), seeds=eval_seeds)
    ev2 = evaluate_policy(net, _boards(stage2, a.smoke), seeds=eval_seeds)

    # 선생님 기준도 학생과 같은 평가 시드로 잰다 — "선생님보다 10점 이내" 가 같은 조건끼리의 비교가 되도록.
    teacher1 = evaluate_teacher(_boards(stage1, a.smoke), seeds=eval_seeds)
    teacher2 = evaluate_teacher(_boards(stage2, a.smoke), seeds=eval_seeds)
    ok = (last["stage1"]["goal_rate"] >= 0.9
          and last["stage1"]["mean_score"] >= teacher1["mean_score"] - 10.0)

    if a.report_only:
        summary_line = (f"- 라운드 {len(rounds)} · 라운드마다 판 {rounds[0]['episodes']}개 · 평가 시드"
                        f" {a.eval_seeds}개 · 이 성적표는 라운드를 다시 안 돌리고 `{a.out}` 의 기존"
                        f" 로그·체크포인트로 다시 만들었다(수집+학습 합계 "
                        f"{sum(r['collect_s'] + r['train']['seconds'] for r in rounds):.0f}초 — "
                        f"원래 실행의 전체 시간은 로그에 없다)")
    else:
        summary_line = (f"- 라운드 {len(rounds)} · 라운드마다 판 {rounds[0]['episodes']}개 · 평가 시드"
                        f" {a.eval_seeds}개 · 전체 {time.perf_counter() - t0:.0f}초")

    d1s, d1t = _distinct_episodes(ev1), _distinct_episodes(teacher1)
    d2s, d2t = _distinct_episodes(ev2), _distinct_episodes(teacher2)

    lines = [
        "# M3 성적표 — DAgger 모방학습", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · 장치 `{dev}` "
        f"· 규칙 스택 `{rs.commit()[:7]}`",
        summary_line,
        f"- 목표: 단계 ① 완주율 ≥ 90%, 점수가 선생님보다 10점 넘게 낮지 않을 것 → "
        f"**{'달성' if ok else '미달'}**",
        f"- 선생님 기준: 단계 ① 완주율 {teacher1['goal_rate']*100:.0f}% 점수 {teacher1['mean_score']:.1f} · "
        f"단계 ② 완주율 {teacher2['goal_rate']*100:.0f}% 점수 {teacher2['mean_score']:.1f}",
        f"- {_teacher_caveat(teacher2)}",
        f"- 서로 다른 판 수(시드 {a.eval_seeds}개 중 실제로 달랐던 것, 마지막 라운드) — 단계 ①: 학생"
        f" {d1s}/{len(ev1['episodes'])}개, 선생님 {d1t}/{len(teacher1['episodes'])}개(액터·신호가 고정이라"
        f" 시드가 달라도 같은 판이 되기 쉽다).",
        f"- 단계 ②: 학생 {d2s}/{len(ev2['episodes'])}개, 선생님 {d2t}/{len(teacher2['episodes'])}개"
        f"(신호 주기 위상이 시드마다 달라 실제로 다른 판이 나온다).", "",
        "| 라운드 | β | 수집 판(완주) | 누적 표본 | 손실 | 정지-출발 표본 | 단계 ① 완주율 | 단계 ① 점수 |"
        " 단계 ② 완주율 | 단계 ② 점수 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rounds:
        lines.append(f"| {r['round']} | {r['beta']} | {r['episodes']}({r['collect_goal']}) | "
                     f"{r['samples']} | {r['train']['loss']:.3f} | {r['stall_start']} | "
                     f"{r['stage1']['goal_rate']*100:.0f}% | {r['stage1']['mean_score']:.1f} | "
                     f"{r['stage2']['goal_rate']*100:.0f}% | {r['stage2']['mean_score']:.1f} |")

    # 가장 좋았던 라운드 — 목표 판정(last)은 그대로 두고, M4 를 위해 "될 수 있었다" 는 사실도 남긴다.
    # 동률이면 더 뒤 라운드(round 값이 더 큰 쪽)를 고른다.
    best = max(rounds, key=lambda r: (r["stage1"]["goal_rate"], r["round"]))
    best_ok = (best["stage1"]["goal_rate"] >= 0.9
              and best["stage1"]["mean_score"] >= teacher1["mean_score"] - 10.0)
    best_ckpt = os.path.join(a.out, f"policy-r{best['round']}.pt")
    lines += ["", f"- 가장 좋았던 라운드: {best['round']}(β={best['beta']}) — 단계 ① 완주율 "
              f"{best['stage1']['goal_rate']*100:.0f}% 점수 {best['stage1']['mean_score']:.1f} → 목표 두 "
              f"조건 **{'달성' if best_ok else '미달'}** · 체크포인트 `{best_ckpt}`"]
    if best["round"] == last["round"] and len(rounds) > 1:
        # 가장 좋았던 라운드가 마지막 라운드와 같으면 "베스트 대 마지막" 비교가 자기 자신과의
        # 비교가 되어 공허하다 — 대신 첫 라운드 대비 얼마나 늘었는지를 보여 준다.
        baseline, baseline_label = rounds[0], f"라운드 {rounds[0]['round']}(첫 라운드)"
    else:
        baseline, baseline_label = best, f"라운드 {best['round']}(가장 좋음)"
    trend = ("늘었다" if last["stall_start"] > baseline["stall_start"]
             else "줄었다" if last["stall_start"] < baseline["stall_start"] else "변하지 않았다")
    lines += [f"- {baseline_label}의 새로 모은 판에서 '정지 상태인데 선생님은 출발하라고 한' 표본이"
              f" {baseline['stall_start']}건, 라운드 {last['round']}(마지막)에서는 "
              f"{last['stall_start']}건이다 — {trend}."]
    lines += _final_outcome_lines(ev1)
    lines += ["", "학생은 결정적 행동으로 혼자 몬다. 점수는 환경이 낸 구간 점수의 평균이고,",
              "환경의 감점표가 대회 채점기와 같다는 것은 M2a·M2b 성적표에 있다.",
              f"산출물(데이터·체크포인트·로그)은 `{a.out}` 아래에 있고 레포에는 넣지 않는다.", "",
              "log_std 바닥은 clamp 로 구현했다 — 바닥에 붙은 파라미터는 그래디언트가 0 이라 스스로 다시",
              "못 올라온다. 이 체크포인트로 M4 의 PPO 를 웜스타트하면 조향 탐색 폭(log_std)이 얼어붙은",
              "채로 시작한다 — softplus 매개변수화나, 매 최적화 스텝 뒤 in-place clamp 로 바꾸는 편이 낫다."]
    lines += _history_lines(a.history, a.history_note)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"rounds": len(rounds), "samples": last["samples"],
                      "stage1": last["stage1"], "stage2": last["stage2"],
                      "teacher1": {k: teacher1[k] for k in ("goal_rate", "mean_score")},
                      "target_met": ok, "report": a.report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
