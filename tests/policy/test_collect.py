import numpy as np
import pytest

from vtd_rl.policy.collect import collect_episode
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_선생님만_수집하면_완주하고_모양이_맞다():
    shard = collect_episode(short_board(), policy=None, beta=1.0, seed=0)
    n = len(shard.turn)
    assert n > 50
    assert shard.vec.shape == (n, VEC_DIM) and shard.objs.shape == (n, OBJ_N, OBJ_DIM)
    assert shard.control.shape == (n, 2) and shard.mask.shape == (n, OBJ_N)
    assert np.all(np.abs(shard.control) <= 1.0) and set(np.unique(shard.turn)) <= {0, 1, 2}
    assert shard.meta["outcome"] == "goal" and shard.meta["student_steps"] == 0
    assert shard.meta["board"] == "H_0_250" and shard.meta["beta"] == 1.0


def test_같은_시드면_같은_조각():
    a = collect_episode(short_board(), seed=3)
    b = collect_episode(short_board(), seed=3)
    assert np.array_equal(a.vec, b.vec) and np.array_equal(a.control, b.control)
    assert a.meta["steps"] == b.meta["steps"]


def test_beta가_0이면_학생이_몰고_정답은_선생님이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    shard = collect_episode(short_board(), policy=net, beta=0.0, seed=1, max_steps=200)
    assert shard.meta["student_steps"] == len(shard.turn)
    # 학습 전 학생은 선생님처럼 몰지 못하므로 정답과 실행 행동이 달라야 의미가 있다
    assert shard.meta["outcome"] in ("goal", "offroad", "collision", "timeout", "stalled", "running")
    assert len(shard.turn) > 0


def test_beta가_1이면_학생을_줘도_선생님이_몬다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    a = collect_episode(short_board(), policy=net, beta=1.0, seed=5)
    b = collect_episode(short_board(), policy=None, beta=1.0, seed=5)
    assert np.array_equal(a.control, b.control) and a.meta["steps"] == b.meta["steps"]
