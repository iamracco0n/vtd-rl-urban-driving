import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.board import Board
from vtd_rl.world.signals import RouteSignal
from vtd_rl.world.world import World, WorldConfig

LANE = {"lane": -1, "w": 3.5, "l": 1.75, "r": 1.75, "pl": 1.75, "pr": 1.75, "xl": 0.0, "xr": 0.0,
        "need": 0.0, "sig": 0, "j": 0, "jx": 0, "lim": 13.9}


def straight_board(length=200, actors=(), duration=60.0, signals="always_green"):
    route = [[float(i), 0.0] for i in range(length + 1)]
    sc = rs.Scenario(name="straight", ego_start=[0.0, 0.0, 0.0], ego_goal=route[-1], speed_limit=13.9,
                     duration=duration, actors=list(actors), lights=[], zones=[], ego_route=route,
                     respawns=[], tl_stops={})
    return Board("straight", sc, [dict(LANE) for _ in route], signals)


def drive(world, steer, accel, seconds):
    info = None
    for _ in range(int(round(seconds / world.cfg.dt))):
        _, info = world.step(steer, accel, 0)
        if info.done:
            break
    return info


def test_리셋():
    w = World(straight_board(), signals=[])
    s = w.reset()
    assert (s.x, s.y, s.heading) == (0.0, 0.0, 0.0)
    assert s.speed == 0.0 and s.tl_id == -1
    assert s.t == pytest.approx(1.0e9) and w.t == 0.0


def test_가속하면_경로를_따라_나아간다():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 3.0)
    assert info.outcome == "running"
    assert 4.0 < info.s < 9.0 and info.v > 3.0
    assert info.t == pytest.approx(3.0)


def test_목표():
    w = World(straight_board(length=40), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 20.0)
    assert info.outcome == "goal" and info.done
    assert info.s >= 40.0 - 10.0


def test_도로_이탈():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.6, 2.0, 20.0)
    assert info.outcome == "offroad"
    assert info.lateral > 1.75 + 1.5


def test_시간_초과():
    w = World(straight_board(duration=1.0), signals=[])
    w.reset()
    assert drive(w, 0.0, 0.0, 5.0).outcome == "timeout"


def test_정체():
    cfg = WorldConfig(stall_seconds=2.0)
    w = World(straight_board(), config=cfg, signals=[])
    w.reset()
    info = drive(w, 0.0, 0.0, 5.0)
    assert info.outcome == "stalled" and info.t == pytest.approx(2.0, abs=0.06)


def test_신호_알림():
    w = World(straight_board(), signals=[RouteSignal(50.0, 7)])
    s = w.reset()
    assert (s.tl_id, s.tl_state) == (7, rs.TL_GREEN)


def test_물체는_80m_안만():
    near = rs.Actor(id=1, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                    motion={"kind": "static", "pos": [40.0, 3.0]})
    far = rs.Actor(id=2, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                   motion={"kind": "static", "pos": [150.0, 0.0]})
    w = World(straight_board(actors=[near, far]), signals=[])
    s = w.reset()
    assert [o.id for o in s.objects] == [1]


def test_리셋하면_처음부터():
    w = World(straight_board(), signals=[])
    w.reset()
    drive(w, 0.0, 2.0, 2.0)
    s = w.reset()
    assert (s.x, w.t, w.ego.v) == (0.0, 0.0, 0.0)
