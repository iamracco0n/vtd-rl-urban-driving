"""M2b 완료 증거 — Gymnasium 검사, 선생님이 환경에서 단계 ① 판 완주, 보상 내역, 걸음 속도.

    env -u PYTHONPATH .venv/bin/python scripts/run_m2b_env.py
    env -u PYTHONPATH .venv/bin/python scripts/run_m2b_env.py --boards course_H --out /tmp/m2b.md
"""
import argparse
import contextlib
import datetime
import os
import platform
import sys
import time

import numpy as np
from gymnasium.utils.env_checker import check_env

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv  # noqa: E402
from vtd_rl.env.teacher_policy import run_teacher_in_env  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402


def random_steps(env, n=200):
    env.reset(seed=0)
    t0 = time.perf_counter()
    steps = 0
    for _ in range(n):
        _o, _r, term, trunc, _i = env.step(env.action_space.sample())
        steps += 1
        if term or trunc:
            env.reset()
    return steps / max(time.perf_counter() - t0, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", nargs="*", default=None, help="판 이름(기본: 단계 ① 전부)")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "reports", "m2b-env.md"))
    a = ap.parse_args()

    _name, boards = load_curriculum(os.path.join(REPO, "curricula", "stage1.json"))
    if a.boards:
        boards = [b for b in boards if b.name in a.boards]
    rows, checked = [], "통과"
    total_steps, total_wall = 0, 0.0
    with contextlib.redirect_stdout(sys.stderr):
        env = VtdDriveEnv(boards, EnvConfig())
        try:
            check_env(env, skip_render_check=True)
        except Exception as exc:                       # noqa: BLE001 — 성적표에 그대로 적는다
            checked = f"실패: {exc}"
        for board in boards:
            t0 = time.perf_counter()
            out = run_teacher_in_env(env, seed=0, options={"board": board.name})
            wall = time.perf_counter() - t0
            sheet = out["info"]["result"]["sheet"]
            hits = ", ".join(f"{i}{'M' if lv == 'major' else 'm'}"
                             for s in sheet for i, lv in sorted(s.items())) or "—"
            rows.append((board.name, out["outcome"], out["steps"], out["sim_time"], out["reward"],
                         out["terms"], hits, out["steps"] / max(wall, 1e-9)))
            total_steps += out["steps"]
            total_wall += wall
        rand_hz = random_steps(env)
        env.close()

    teacher_hz = total_steps / max(total_wall, 1e-9)
    lines = [
        "# M2b 성적표 — Gymnasium 환경", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · 규칙 스택 `{rs.commit()[:7]}`",
        f"- `gymnasium.utils.env_checker.check_env`: **{checked}**",
        "- 한 걸음 = 판단 10 Hz(시뮬 두 프레임). 심판과 행 기록은 프레임마다 돈다",
        f"- 선생님 걸음 속도: {teacher_hz:.0f} 걸음/s ({1e6 / teacher_hz:.0f} µs/걸음)",
        f"- 무작위 행동 걸음 속도: {rand_hz:.0f} 걸음/s ({1e6 / rand_hz:.0f} µs/걸음)", "",
        "| 판 | 결과 | 걸음 | 시뮬 시간[s] | 보상 합 | 진행 | 위반 | 승차감 | 심판 감점 | 걸음/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, outcome, steps, sim, reward, terms, hits, hz in rows:
        lines.append(f"| {name} | {outcome} | {steps} | {sim:.1f} | {reward:.1f} | "
                     f"{terms.get('progress', 0.0):.1f} | {terms.get('violation', 0.0):.1f} | "
                     f"{terms.get('comfort', 0.0):.1f} | {hits} | {hz:.0f} |")
    lines += ["", "선생님이 환경에서 단계 ① 판을 완주하는지, 그때 보상이 어떻게 쌓이는지가 M2b 의 증거다.",
              "판정 자체의 채점기 일치는 M2a 성적표(docs/reports/m2a-referee-parity.md)에 있다."]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(a.out)
    return 0 if checked == "통과" and all(r[1] == "goal" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
