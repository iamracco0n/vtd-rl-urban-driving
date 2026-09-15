"""선생님이 단계 ① 판을 전부 달리고 성적표를 쓴다 — M1 완료 증거.

    .venv/bin/python scripts/run_stage1_teacher.py --bench '<bench_world.py 출력 JSON>'
CSV 는 runs/m1/ 에 남긴다(git 제외).
"""
import argparse
import datetime
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.rollout import run_teacher_episode  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="scripts/bench_world.py 가 출력한 JSON 한 줄")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "reports", "m1-stage1-teacher.md"))
    a = ap.parse_args()
    bench = json.loads(a.bench)
    name, boards = load_curriculum(os.path.join(REPO, "curricula", "stage1.json"))
    os.makedirs(os.path.join(REPO, "runs", "m1"), exist_ok=True)

    rows = []
    for b in boards:
        r = run_teacher_episode(b, log_csv=os.path.join(REPO, "runs", "m1", f"{b.name}.csv"))
        rows.append((b, r))
        print(f"{b.name}: {r.outcome} {r.distance:.0f}/{b.route.total:.0f} m "
              f"{r.sim_time:.1f}s {r.steps_per_sec:.0f} steps/s", flush=True)

    ok = sum(1 for _, r in rows if r.outcome == "goal")
    lines = [
        "# M1 성적표 — 선생님(규칙 스택)이 오프라인 세계에서 단계 ① 판을 달린 결과", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · Python {platform.python_version()}",
        f"- 규칙 스택 커밋 `{rs.commit()}`",
        f"- 단계 ① `{name}` · 판 {len(rows)}개 · **완주 {ok}/{len(rows)}**", "",
        "| 판 | 경로 길이 [m] | 결과 | 달린 거리 [m] | 시뮬 시간 [s] | 스텝 | 벽시계 [s] | 스텝/초 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for b, r in rows:
        lines.append(f"| {b.name} | {b.route.total:.0f} | {r.outcome} | {r.distance:.0f} | {r.sim_time:.1f} "
                     f"| {r.steps} | {r.wall_time:.1f} | {r.steps_per_sec:.0f} |")
    lines += [
        "", "## 스텝 속도(`scripts/bench_world.py`)", "",
        f"- 판 `{bench['board']}` · 시뮬 {bench['sim_seconds']:.0f} 초",
        f"- world 단독 **{bench['world_steps_per_sec']} 스텝/초**",
        f"- world + 선생님 **{bench['world_plus_teacher_steps_per_sec']} 스텝/초**", "",
        "동역학 지연·조향 속도는 M5 보정 전 초기값(`DynamicsParams` 기본값)이다.",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"성적표: {a.out}")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
