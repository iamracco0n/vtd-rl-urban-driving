"""판×시드마다 조각 하나를 모은다(여러 프로세스).

    env -u PYTHONPATH .venv/bin/python scripts/collect_episodes.py \
        --curriculum curricula/stage1.json --out runs/lab-main/dagger/data --round 0 \
        --seeds 2 --beta 1.0 --workers 12
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.policy.collect import collect_episode  # noqa: E402
from vtd_rl.policy.dataset import save_shard  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def _one(job):
    curriculum, name, seed, beta, policy_path, out, rnd = job
    _n, boards = load_curriculum(curriculum)
    board = next(b for b in boards if b.name == name)
    policy = DrivePolicy.load(policy_path) if policy_path else None
    shard = collect_episode(board, policy=policy, beta=beta, seed=seed)
    path = os.path.join(out, f"r{rnd}-{name}-s{seed}.npz")
    save_shard(shard, path)
    return {"path": path, **shard.meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()

    _name, boards = load_curriculum(a.curriculum)
    jobs = [(a.curriculum, b.name, 1000 * a.round + s, a.beta, a.policy, a.out, a.round)
            for b in boards for s in range(a.seeds)]
    os.makedirs(a.out, exist_ok=True)
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            metas = pool.map(_one, jobs)
    else:
        metas = [_one(j) for j in jobs]
    print(json.dumps({"round": a.round, "episodes": len(metas),
                      "samples": sum(m["steps"] for m in metas),
                      "goal": sum(1 for m in metas if m["outcome"] == "goal"),
                      "shards": metas}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
