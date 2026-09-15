import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.actors import ActorSet, nearest_objects


def actor(aid, spawn, motion, size=(0.15, 0.46, 0.61), typ="object"):
    return rs.Actor(id=aid, type=typ, size=list(size), spawn=spawn, motion=motion)


def test_정지_물체():
    a = actor(401, {"at_time": 0}, {"kind": "static", "pos": [1474.915, -548.598]})
    objs = ActorSet([a]).step(0.0, 1470.0, -550.0, 0.05)
    assert len(objs) == 1
    o = objs[0]
    assert (o.id, o.x, o.y, o.speed) == (401, pytest.approx(1474.915), pytest.approx(-548.598), 0.0)
    assert (o.length, o.width, o.height) == (0.15, 0.46, 0.61)


def test_등장_트리거와_리셋():
    a = actor(7, {"ego_within": 20, "of": [100.0, 0.0]}, {"kind": "static", "pos": [100.0, 3.0]})
    acts = ActorSet([a])
    assert acts.step(0.0, 0.0, 0.0, 0.05) == []
    assert len(acts.step(0.05, 85.0, 0.0, 0.05)) == 1
    assert len(acts.step(0.10, 0.0, 0.0, 0.05)) == 1       # 한 번 등장하면 남는다
    acts.reset()
    assert acts.step(0.0, 0.0, 0.0, 0.05) == []


def test_횡단_보행자는_움직인다():
    a = actor(9, {"at_time": 0}, {"kind": "crossing", "pos": [50.0, -5.0], "vel": [0.0, 1.5]},
              size=(0.6, 0.7, 1.8), typ="pedestrian")
    acts = ActorSet([a])
    for k in range(20):
        objs = acts.step(k * 0.05, 0.0, 0.0, 0.05)
    assert objs[0].y == pytest.approx(-5.0 + 1.5 * 1.0, abs=0.01)


def test_가까운_순_거르기():
    objs = [rs.Obj(i, float(d), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) for i, d in ((1, 50), (2, 10), (3, 90), (4, 30))]
    out = nearest_objects(objs, 0.0, 0.0, max_range=80.0, max_n=2)
    assert [o.id for o in out] == [2, 4]
