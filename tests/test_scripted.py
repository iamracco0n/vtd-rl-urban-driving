import math

import pytest

from vtd_rl.drivers.scripted import Cruise, LaneShift, Offset, ScriptedDriver, SignalWindow, StopAt
from vtd_rl.rollout import run_episode, run_teacher_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_일정():
    assert Cruise(5.0)(10.0, 0.0) == 5.0
    st = StopAt(50.0, hold=2.0, v=8.0)
    assert st(0.0, 0.0) == 8.0
    assert st(45.0, 1.0) == pytest.approx(math.sqrt(2 * 2.0 * 5.0))
    assert st(49.8, 3.0) == 0.0
    assert st(49.9, 4.9) == 0.0
    assert st(49.9, 5.1) == 8.0
    assert st(60.0, 6.0) == 8.0
    ls = LaneShift(100.0, delta=3.2, length=40.0)
    assert ls(90.0, 0) == 0.0 and ls(120.0, 0) == pytest.approx(1.6) and ls(200.0, 0) == 3.2
    sw = SignalWindow(10.0, 20.0, side=1)
    assert (sw(5.0, 0), sw(15.0, 0), sw(25.0, 0)) == (0, 1, 0)
    assert Offset(1.5)(0.0, 0.0) == 1.5


def test_대본_운전자는_경로를_따라_완주():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, Cruise(8.0), Offset(0.0), SignalWindow(0, 0, 0)))
    assert run.result.outcome == "goal"
    assert len(run.rows) == run.result.steps
    assert max(abs(r["d_ego"]) for r in run.rows[100:]) < 1.5
    assert {r["reason"] for r in run.rows} == {"SCRIPT"}
    assert max(r["v"] for r in run.rows) == pytest.approx(8.0, abs=1.0)


def test_정지_일정을_따른다():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, StopAt(60.0, 4.0, 8.0), Offset(0.0), SignalWindow(0, 0, 0)))
    stopped = [r for r in run.rows if r["t"] > 2.0 and r["v"] <= 1 / 3.6]
    assert stopped and stopped[-1]["t"] - stopped[0]["t"] > 2.5
    assert run.result.outcome == "goal"


def test_횡오프셋과_지시등을_따른다():
    b = h_slice()
    run = run_episode(b, ScriptedDriver(b, Cruise(6.0), LaneShift(150.0, 1.2, 20.0),
                                        SignalWindow(140.0, 180.0, 1)))
    late = [r["d_ego"] for r in run.rows if r["t"] > 32.0]
    assert late and min(late) > 0.8
    assert any(r["sig"] == 1 for r in run.rows) and run.rows[-1]["sig"] == 0


def test_선생님_래퍼는_그대로():
    assert run_teacher_episode(h_slice()).outcome == "goal"
