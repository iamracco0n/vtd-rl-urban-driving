import os

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.teacher_policy import TeacherPolicy, run_teacher_in_env
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, load_curriculum, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "..", "curricula", "stage1.json")


def test_선생님은_환경에서도_판을_끝낸다():
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    env = VtdDriveEnv([b])
    out = run_teacher_in_env(env, seed=0)
    assert out["outcome"] == "goal"
    assert out["terms"]["goal"] > 0.0
    assert out["reward"] > 0.0
    env.close()


def test_선생님은_매_프레임_돈다():
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    env = VtdDriveEnv([b])
    policy = TeacherPolicy(env)
    env.reset(seed=0)
    policy.reset()
    seen = []
    inner = env.frame_hook
    env.frame_hook = lambda state, clock: (seen.append(clock), inner(state, clock))[1]
    for _ in range(5):
        env.step(policy.act())
    assert len(seen) == 10                      # 걸음 5 × 프레임 2
    assert seen == sorted(seen)
    env.close()


@pytest.mark.slow
def test_환경이_남긴_CSV_를_채점기로_매기면_심판과_같다(tmp_path):
    _name, boards = load_curriculum(STAGE1)
    env = VtdDriveEnv(boards, EnvConfig(log_dir=str(tmp_path)))
    for board in boards:
        out = run_teacher_in_env(env, seed=0, options={"board": board.name})
        assert out["outcome"] == "goal", board.name
        rows = rs.score_fma.load(out["info"]["result"]["rows_csv"])
        oracle = score_episode(rows, board, sections=5, use_map=True)
        assert [dict(s) for s in oracle.state] == out["info"]["result"]["sheet"], board.name
        assert oracle.respawns == out["info"]["result"]["respawns"], board.name
    env.close()
