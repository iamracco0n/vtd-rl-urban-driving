import math

import numpy as np
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.action import ActionConfig, action_space, frames_per_step, from_command, to_command


def test_변환():
    cfg = ActionConfig()
    steer, accel, turn = to_command({"control": np.array([1.0, 1.0], np.float32), "turn": 1}, cfg)
    assert steer == pytest.approx(math.radians(35)) and accel == pytest.approx(2.0)
    assert turn == rs.TS_LEFT
    steer, accel, turn = to_command({"control": np.array([-0.5, -1.0], np.float32), "turn": 0}, cfg)
    assert steer == pytest.approx(-math.radians(35) / 2) and accel == pytest.approx(-5.0)
    assert turn == rs.TS_OFF
    _s, accel, _t = to_command({"control": np.array([0.0, 0.0], np.float32), "turn": 0}, cfg)
    assert accel == 0.0


def test_범위를_벗어난_행동은_자른다():
    cfg = ActionConfig()
    steer, accel, _t = to_command({"control": np.array([9.0, -9.0], np.float32), "turn": 2}, cfg)
    assert steer == pytest.approx(math.radians(35)) and accel == pytest.approx(-5.0)


def test_역변환은_왕복한다():
    cfg = ActionConfig()
    for steer, accel, turn in [(0.1, 1.0, rs.TS_OFF), (-0.3, -2.0, rs.TS_RIGHT), (0.0, 0.0, rs.TS_LEFT)]:
        a = from_command(steer, accel, turn, cfg)
        assert action_space(cfg).contains(a)
        s2, a2, t2 = to_command(a, cfg)
        assert (s2, t2) == (pytest.approx(steer), turn) and a2 == pytest.approx(accel)


def test_판단_주기():
    assert frames_per_step(ActionConfig(), 0.05) == 2
    assert frames_per_step(ActionConfig(decision_hz=20.0), 0.05) == 1
    assert frames_per_step(ActionConfig(decision_hz=5.0), 0.05) == 4
