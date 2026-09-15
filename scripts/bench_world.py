"""world 단독·선생님 포함 스텝 속도를 잰다.

    .venv/bin/python scripts/bench_world.py --board course_H --seconds 120
"""
import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vtd_rl.teacher.shadow import ShadowTeacher  # noqa: E402
from vtd_rl.world.board import load_board  # noqa: E402
from vtd_rl.world.world import World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="course_H")
    ap.add_argument("--seconds", type=float, default=120.0, help="시뮬 시간[s]")
    a = ap.parse_args()

    # 규칙 스택(third_party)이 판에 따라 stdout 에 print 를 남길 수 있다(예: course_H
    # 차로계획 재정렬 알림). 우리 stdout 은 JSON 한 줄만이어야 호출부(run_stage1_teacher.py
    # 의 `B=$(...)` 캡처)가 안전하다 — 그런 출력은 전부 stderr 로 돌린다.
    with contextlib.redirect_stdout(sys.stderr):
        letter = a.board.split("_")[-1]
        board = load_board({"name": a.board, "route": f"routes/HL_FMA_NEW_{letter}.json",
                            "lane": f"routes/HL_FMA_NEW_{letter}_lane.json"})
        n = int(a.seconds / 0.05)

        w = World(board)
        w.reset()
        t0 = time.perf_counter()
        for k in range(n):
            _, info = w.step(0.0, 1.0 if w.ego.v < 8.0 else 0.0, 0)
            if info.done:
                w.reset()
        world_sps = n / (time.perf_counter() - t0)

        w = World(board)
        s = w.reset()
        teacher = ShadowTeacher(board)
        t0 = time.perf_counter()
        for k in range(n):
            cmd = teacher.act(s, w.clock)
            s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
            if info.done:
                break
        both_sps = (k + 1) / (time.perf_counter() - t0)

    print(json.dumps({"board": a.board, "sim_seconds": a.seconds,
                      "world_steps_per_sec": round(world_sps),
                      "world_plus_teacher_steps_per_sec": round(both_sps)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
