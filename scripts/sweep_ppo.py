"""같은 설정을 시드 여러 개로 돌려 실행 간 편차를 잰다.

    env -u PYTHONPATH .venv/bin/python scripts/sweep_ppo.py \
        --out-root runs/omen/$(date +%F)-sweep --seeds 0 1 2 --name m4a-baseline \
        -- --init runs/lab-main/.../policy-r4.pt --dagger-data runs/lab-main/.../data
    env -u PYTHONPATH .venv/bin/python scripts/sweep_ppo.py \
        --out-root /tmp/sweep --name smoke --seeds 0 1 -- --smoke

M4a 는 설정마다 시드 하나였다. M3 에서 같은 명령이 완주율 0% 와 100% 로 갈린 전례가 있어
(docs/reports/m3-dagger.md), 한 번의 숫자가 편차 안인지 밖인지 알 수 없었다.
시드는 **순차로** 돈다 — 30 환경짜리 실행 둘을 동시에 띄우면 코어가 경합한다.

`--` 뒤의 인자는 그대로 `train_ppo.py` 로 넘어간다(`--out`·`--seed` 는 이 스크립트가 시드마다
직접 채우므로 뒤쪽에 다시 주면 안 된다 — argparse 가 중복을 그대로 받아 뒤엣것이 이기므로
조용히 잘못된 값으로 덮인다).
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_ppo.py")


def _spread(values):
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="`--` 뒤의 인자를 train_ppo.py 로 그대로 넘긴다")
    a = ap.parse_args()
    extra = [x for x in a.rest if x != "--"]

    runs = []
    for seed in a.seeds:
        out = os.path.join(a.out_root, f"{a.name}-s{seed}")
        cmd = [sys.executable, TRAIN, "--out", out, "--seed", str(seed)] + extra
        # 시드는 순차로 돈다 — 여기서 asyncio/멀티프로세싱으로 동시에 띄우면 각 실행이 요구하는
        # 벡터 환경(기본 cpu_count-2 개)이 서로 코어를 나눠 먹어 실행 시간도 늘고 결과도 오염된다.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            sys.exit(f"시드 {seed} 실행이 실패했다(rc={proc.returncode}) — 여기서 멈춘다")
        runs.append({"seed": seed, "summary": json.loads(proc.stdout.strip().splitlines()[-1])})

    spread = {}
    for stage in ("stage1", "stage2"):
        spread[stage] = {k: _spread([r["summary"]["stages"][stage][k] for r in runs])
                         for k in ("goal_rate", "mean_score")}
    summary = {"name": a.name, "seeds": a.seeds, "runs": runs, "spread": spread}
    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
