import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def boards():
    b = load_board(H)
    return [slice_board(b, 0.0, 250.0, "H_0_250"), slice_board(b, 250.0, 500.0, "H_250_500")]


def test_gymnasium_환경_검사():
    env = VtdDriveEnv([boards()[0]])
    check_env(env, skip_render_check=True)
    env.close()


def test_리셋은_시드로_재현된다():
    env = VtdDriveEnv(boards())
    o1, i1 = env.reset(seed=3)
    o2, i2 = env.reset(seed=3)
    assert i1["board"] == i2["board"]
    for k in o1:
        assert np.array_equal(o1[k], o2[k])
    picked = {env.reset(seed=s)[1]["board"] for s in range(12)}
    assert len(picked) == 2                      # 판을 섞어 고른다
    o3, i3 = env.reset(seed=3, options={"board": "H_250_500"})
    assert i3["board"] == "H_250_500"
    env.close()


def test_한_걸음은_두_프레임이고_심판이_따라온다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        {"control": np.array([0.0, 1.0], np.float32), "turn": 0})
    assert info["frames"] == 2
    assert env.world.t == pytest.approx(0.1)
    assert not terminated and not truncated
    assert info["reward_terms"]["progress"] >= 0.0
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert env.referee.ctx.secs is not None
    env.close()


def test_과속하면_위반_보상이_깎인다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    worst, items = 0.0, set()
    for _ in range(300):
        _o, r, term, trunc, info = env.step({"control": np.array([0.0, 1.0], np.float32), "turn": 0})
        worst = min(worst, info["reward_terms"]["violation"])
        items |= {h[2] for h in info["hits"]}
        if term or trunc:
            break
    assert worst < 0.0 and 1 in items            # 제한 50 인데 계속 가속하면 항목 1
    env.close()


def test_판이_끝나면_성적이_따라온다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    for _ in range(2000):
        _o, _r, term, trunc, info = env.step({"control": np.array([0.0, 0.6], np.float32), "turn": 0})
        if term or trunc:
            break
    assert info["outcome"] in ("goal", "offroad", "timeout", "stalled", "collision")
    assert "result" in info and "sheet" in info["result"]
    assert len(info["result"]["sheet"]) == 5
    env.close()


def test_행_CSV_남기기(tmp_path):
    env = VtdDriveEnv([boards()[0]], EnvConfig(log_dir=str(tmp_path)))
    env.reset(seed=0)
    for _ in range(5):
        env.step({"control": np.array([0.0, 0.5], np.float32), "turn": 0})
    env.reset(seed=1)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files and files[0].startswith("H_0_250-")
    lines = open(tmp_path / files[0], encoding="utf-8").readlines()
    assert lines[0].startswith("t,x,y,heading,v")
    assert len(lines) == 5 * 2 + 1        # 머리글 + 시뮬 프레임마다 한 행(걸음마다 두 프레임)
    env.close()


def test_frame_hook은_시뮬_프레임마다_불린다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    clocks = []
    env.frame_hook = lambda state, clock: clocks.append(clock)
    n_steps = 5
    for _ in range(n_steps):
        _o, _r, term, trunc, _info = env.step(
            {"control": np.array([0.0, 0.5], np.float32), "turn": 0})
        assert not term and not trunc
    assert len(clocks) == 2 * n_steps
    assert all(a <= b for a, b in zip(clocks, clocks[1:]))
    env.close()


def test_종료_후_다시_밟아도_성적표가_그대로다():
    env = VtdDriveEnv([boards()[0]])
    env.reset(seed=0)
    info = None
    for _ in range(60):
        _o, _r, term, trunc, info = env.step(
            {"control": np.array([1.0, 0.6], np.float32), "turn": 0})   # 최대 조향으로 도로 이탈 유도
        if term or trunc:
            break
    assert term or trunc
    assert "result" in info
    sheet_before = [dict(s) for s in env.referee.sheet.state]

    calls = []
    env.frame_hook = lambda state, clock: calls.append(clock)
    _o2, reward2, term2, trunc2, info2 = env.step(
        {"control": np.array([0.0, 0.0], np.float32), "turn": 0})

    assert [dict(s) for s in env.referee.sheet.state] == sheet_before
    assert "result" not in info2
    assert reward2 == 0.0
    assert calls == []                    # 이미 끝난 판은 프레임을 더 밟지 않는다
    assert (term2, trunc2) == (term, trunc)
    env.close()


def test_리셋_전에_스텝하면_명확한_오류():
    env = VtdDriveEnv([boards()[0]])
    with pytest.raises(RuntimeError):
        env.step({"control": np.array([0.0, 0.0], np.float32), "turn": 0})
    env.close()


def test_없는_판_이름은_명확한_오류():
    env = VtdDriveEnv([boards()[0]])
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"board": "없는판"})
    env.close()
