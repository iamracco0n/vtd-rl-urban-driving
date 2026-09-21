"""같은 설정을 시드 여러 개로 돌려 실행 간 편차를 잰다.

    env -u PYTHONPATH .venv/bin/python scripts/sweep_ppo.py \
        --out-root runs/omen/$(date +%F)-sweep --seeds 0 1 2 --name m4a-baseline \
        -- --init runs/lab-main/.../policy-r4.pt --dagger-data runs/lab-main/.../data
    env -u PYTHONPATH .venv/bin/python scripts/sweep_ppo.py \
        --out-root /tmp/sweep --name smoke --seeds 0 1 -- --smoke

M4a 는 설정마다 시드 하나였다. M3 에서 같은 명령이 완주율 0% 와 100% 로 갈린 전례가 있어
(docs/reports/m3-dagger.md), 한 번의 숫자가 편차 안인지 밖인지 알 수 없었다.
시드는 **순차로** 돈다 — 30 환경짜리 실행 둘을 동시에 띄우면 코어가 경합한다.

`--` 뒤의 인자는 그대로 `train_ppo.py` 로 넘어간다. `--out`·`--seed` 는 이 스크립트가 시드마다
직접 채우므로 뒤쪽에 다시 주면 안 된다 — 예전엔 문서로만 경고했는데, argparse 는 중복을 그대로
받아 뒤엣것이 조용히 이겨(세 시드가 전부 같은 값으로 도는데 에러가 안 남) 스윕의 목적 자체를
무효화한다(2026-09-21 리뷰 지적). 지금은 `extra` 에 이 둘이 보이면 `ap.error` 로 막는다.

시드마다 성공하는 즉시 `<out-root>/<name>-s<seed>/summary.json` 을 쓰고 `<out-root>/sweep.json`
을 그때까지의 결과로 다시 쓴다(`"complete": false`) — 9 개 중 8 번째가 죽어도 앞선 7 개의
cross-seed 통계(스프레드)가 파이썬 지역 변수 속에서만 살다 사라지지 않는다(2026-09-21 리뷰
지적: 원래 스켈레톤은 루프가 끝까지 성공해야만 sweep.json 을 썼다). 부분 실패 뒤 같은 명령을
그냥 다시 돌리면 이미 끝난 시드가 재사용 가드에 걸려 또 죽으므로, `--resume` 을 주면 이미
`summary.json` 이 있는 시드는 다시 안 돌리고 그 파일을 읽어 쓴다.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_ppo.py")


def _spread(values):
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _forbidden_flag(extra):
    """`extra` 에 `--seed`/`--out` 이 섞여 있으면 그 플래그 이름을 반환한다(없으면 None).

    `--seed=5`/`--out=x` 형태도 잡는다. `--out-root` 는 이 스크립트 자신의 인자라 별개이고
    `--out` 으로 시작은 하지만 `--out` 도 `--out=...` 도 아니므로 여기 걸리지 않는다.
    """
    for tok in extra:
        for flag in ("--seed", "--out"):
            if tok == flag or tok.startswith(flag + "="):
                return flag
    return None


def _write_sweep_json(out_root, name, seeds, runs, complete):
    """지금까지 모은 `runs` 로 `spread` 를 다시 계산해 `sweep.json` 을 (다시) 쓴다.

    실행이 하나만 있어도 `spread` 는 계산할 수 있다(min==max==mean) — 다만 `complete`가
    거짓이면 아직 다 안 돈 부분 결과라는 뜻이니, 읽는 쪽이 완성본으로 착각하지 않도록 그
    플래그를 같이 남긴다.
    """
    spread = {}
    if runs:
        for stage in ("stage1", "stage2"):
            spread[stage] = {k: _spread([r["summary"]["stages"][stage][k] for r in runs])
                             for k in ("goal_rate", "mean_score")}
    summary = {"name": name, "seeds": seeds, "runs": runs, "spread": spread, "complete": complete}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def _stage_line(summary: dict) -> str:
    return " ".join(f"{label}={ev['goal_rate']*100:.0f}%/{ev['mean_score']:.1f}"
                    for label, ev in summary["stages"].items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="시드 폴더에 이미 summary.json 이 있으면 다시 안 돌리고 그걸 읽는다"
                        "(없으면 지금처럼 크게 실패한다 — 조용한 오염 방지)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="`--` 뒤의 인자를 train_ppo.py 로 그대로 넘긴다")
    a = ap.parse_args()
    extra = [x for x in a.rest if x != "--"]
    bad_flag = _forbidden_flag(extra)
    if bad_flag:
        ap.error(f"`{bad_flag}` 는 이 스크립트가 시드마다 직접 채운다 — `--` 뒤에 다시 주면 "
                f"안 된다(뒤엣것이 조용히 이겨 모든 시드가 같은 값으로 돌아도 에러가 안 난다).")

    runs = []
    for seed in a.seeds:
        out = os.path.join(a.out_root, f"{a.name}-s{seed}")
        summary_path = os.path.join(out, "summary.json")
        t0 = time.perf_counter()
        if a.resume and os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            sys.stderr.write(f"[sweep] 시드 {seed} — 이미 끝남, {summary_path} 재사용\n")
        else:
            sys.stderr.write(f"[sweep] 시드 {seed} 시작 → {out}\n")
            cmd = [sys.executable, TRAIN, "--out", out, "--seed", str(seed)] + extra
            # 시드는 순차로 돈다 — 여기서 asyncio/멀티프로세싱으로 동시에 띄우면 각 실행이
            # 요구하는 벡터 환경(기본 cpu_count-2 개)이 서로 코어를 나눠 먹어 실행 시간도
            # 늘고 결과도 오염된다.
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr[-4000:])
                sys.exit(f"시드 {seed} 실행이 실패했다(rc={proc.returncode}) — 여기서 멈춘다"
                        f"(먼저 끝난 시드는 {a.out_root}/{a.name}-s<seed>/summary.json 으로 "
                        "남아 있다 — --resume 으로 이어서 돌릴 수 있다)")
            summary = json.loads(proc.stdout.strip().splitlines()[-1])
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=1)
        elapsed = time.perf_counter() - t0
        sys.stderr.write(f"[sweep] 시드 {seed} 끝 — {elapsed:.0f}초, {_stage_line(summary)}\n")
        runs.append({"seed": seed, "summary": summary})
        # 시드가 하나 끝날 때마다 즉시 sweep.json 을 (지금까지 모은 것으로) 다시 쓴다 — 다음
        # 시드에서 죽어도 이 시드까지의 cross-seed 스프레드가 파이썬 지역 변수 속에서만 살다
        # 사라지지 않는다(2026-09-21 리뷰 지적: 원래는 루프가 끝까지 성공해야만 파일이 생겼다).
        _write_sweep_json(a.out_root, a.name, a.seeds, runs, complete=False)

    final = _write_sweep_json(a.out_root, a.name, a.seeds, runs, complete=True)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
