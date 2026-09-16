"""M3 — DAgger 라운드 반복과 성적표.

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --out runs/lab-main/$(date +%F)-dagger --rounds 5 --seeds 2 --workers 12
    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --smoke --out /tmp/smoke
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
from vtd_rl.policy.dataset import load_dir, save_shard  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher  # noqa: E402
from vtd_rl.policy.net import DrivePolicy, PolicyConfig  # noqa: E402
from vtd_rl.policy.train import TrainConfig, train_epochs  # noqa: E402
from vtd_rl.world.board import load_board, load_curriculum, slice_board  # noqa: E402

BETAS = [1.0, 0.5, 0.25, 0.1, 0.0]
H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


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
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.seeds, a.epochs, a.eval_seeds, a.workers = 1, 1, 2, 1, 1

    data_dir = os.path.join(a.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(a.out, "log.jsonl")
    dev = pick_device(a.device)
    net = DrivePolicy(PolicyConfig(trunk=(64, 64)) if a.smoke else PolicyConfig()).to(dev)
    stage1 = os.path.join(REPO, "curricula", "stage1.json")
    stage2 = os.path.join(REPO, "curricula", "stage2.json")
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

        dataset = load_dir(data_dir)
        train = train_epochs(net, dataset, TrainConfig(epochs=a.epochs, seed=rnd), device=dev)
        net.save(os.path.join(a.out, f"policy-r{rnd}.pt"))

        seeds = tuple(range(a.eval_seeds))
        ev1 = evaluate_policy(net, _boards(stage1, a.smoke), seeds=seeds)
        ev2 = evaluate_policy(net, _boards(stage2, a.smoke), seeds=seeds)
        row = {"round": rnd, "beta": beta, "episodes": len(metas),
               "collect_goal": sum(1 for m in metas if m["outcome"] == "goal"),
               "samples": len(dataset), "collect_s": collect_s, "train": train,
               "stage1": {k: ev1[k] for k in ("goal_rate", "mean_score", "mean_reward")},
               "stage2": {k: ev2[k] for k in ("goal_rate", "mean_score", "mean_reward")}}
        rounds.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    teacher1 = evaluate_teacher(_boards(stage1, a.smoke), seeds=(0,))
    teacher2 = evaluate_teacher(_boards(stage2, a.smoke), seeds=(0,))
    last = rounds[-1]
    ok = (last["stage1"]["goal_rate"] >= 0.9
          and last["stage1"]["mean_score"] >= teacher1["mean_score"] - 10.0)

    lines = [
        "# M3 성적표 — DAgger 모방학습", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · 장치 `{dev}` "
        f"· 규칙 스택 `{rs.commit()[:7]}`",
        f"- 라운드 {a.rounds} · 라운드마다 판 {rounds[0]['episodes']}개 · 평가 시드 {a.eval_seeds}개"
        f" · 전체 {time.perf_counter() - t0:.0f}초",
        f"- 목표: 단계 ① 완주율 ≥ 90%, 점수가 선생님보다 10점 넘게 낮지 않을 것 → "
        f"**{'달성' if ok else '미달'}**",
        f"- 선생님 기준: 단계 ① 완주율 {teacher1['goal_rate']*100:.0f}% 점수 {teacher1['mean_score']:.1f} · "
        f"단계 ② 완주율 {teacher2['goal_rate']*100:.0f}% 점수 {teacher2['mean_score']:.1f}", "",
        "| 라운드 | β | 수집 판(완주) | 누적 표본 | 손실 | 단계 ① 완주율 | 단계 ① 점수 | 단계 ② 완주율 | 단계 ② 점수 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rounds:
        lines.append(f"| {r['round']} | {r['beta']} | {r['episodes']}({r['collect_goal']}) | "
                     f"{r['samples']} | {r['train']['loss']:.3f} | "
                     f"{r['stage1']['goal_rate']*100:.0f}% | {r['stage1']['mean_score']:.1f} | "
                     f"{r['stage2']['goal_rate']*100:.0f}% | {r['stage2']['mean_score']:.1f} |")
    lines += ["", "학생은 결정적 행동으로 혼자 몬다. 점수는 환경이 낸 구간 점수의 평균이고,",
              "환경의 감점표가 대회 채점기와 같다는 것은 M2a·M2b 성적표에 있다.",
              f"산출물(데이터·체크포인트·로그)은 `{a.out}` 아래에 있고 레포에는 넣지 않는다."]
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
