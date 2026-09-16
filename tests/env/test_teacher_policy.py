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


def _run(env, policy, seed=0, options=None, max_steps=20000):
    env.reset(seed=seed, options=options)
    total, steps = 0.0, 0
    for _ in range(max_steps):
        _o, r, term, trunc, info = env.step(policy.act())
        total += r
        steps += 1
        if term or trunc:
            break
    return {"outcome": info["outcome"], "steps": steps, "reward": total,
            "sheet": info["result"]["sheet"]}


def test_같은_정책을_두_판에_다시_써도_결과가_같다():
    """`ShadowTeacher` 는 추월 상태기계와 `_last_now` 를 들고 있다 — 판이 바뀌면 새로 지어야 한다.

    판 이름만 보고 재사용하던 때의 실측: 219 -> 221 걸음, 보상 143.6554 -> 143.5132.
    (`_last_now` 가 지난 판의 끝 시각이라 첫 프레임 `dt` 가 1e-3 바닥으로 무너졌다.)
    """
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    env = VtdDriveEnv([b])
    reused = TeacherPolicy(env)
    first, second = _run(env, reused), _run(env, reused)
    fresh = _run(env, TeacherPolicy(env))
    env.close()
    for k in ("outcome", "steps", "reward", "sheet"):
        assert second[k] == first[k], k
        assert fresh[k] == first[k], k


def test_선생님이_물러나면_갈고리도_뗀다():
    """선생님 판이 끝나면 frame_hook·command_tags 가 남지 않는다 — 학생 판에 규칙 스택이 따라붙지 않게."""
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    env = VtdDriveEnv([b])
    run_teacher_in_env(env, seed=0)
    assert env.frame_hook is None and env.command_tags is None
    env.close()


def test_같은_판을_다시_돌려도_결과가_같다():
    """판마다 세계·심판·보상이 깨끗이 초기화되는가(세계는 캐시에서 재사용된다)."""
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
    env = VtdDriveEnv([b])
    first = run_teacher_in_env(env, seed=0)
    second = run_teacher_in_env(env, seed=0)
    env.close()
    for k in ("outcome", "steps", "reward"):
        assert second[k] == first[k], k
    assert second["info"]["result"]["sheet"] == first["info"]["result"]["sheet"]


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
