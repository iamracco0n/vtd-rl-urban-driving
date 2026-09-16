import math

import numpy as np
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.observation import ObsConfig, build_observation, observation_space
from vtd_rl.world.board import load_board, slice_board
from vtd_rl.world.world import World

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_world(signals="always_green"):
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250", signals=signals)
    return World(b)


def test_관측은_공간_안에_들어간다():
    w = h_world()
    state = w.reset()
    info = None
    space = observation_space()
    for k in range(40):
        obs = build_observation(w, state, info, (0.1, 0.5, rs.TS_LEFT))
        assert space.contains(obs), {k: (v.dtype, float(v.min()), float(v.max()))
                                     for k, v in obs.items()}
        state, info = w.step(0.0, 1.0, rs.TS_LEFT)


def test_자차_묶음():
    w = h_world()
    state = w.reset()
    obs = build_observation(w, state, None, (0.0, 0.0, rs.TS_OFF))
    assert obs["ego"][0] == pytest.approx(0.0)              # 정지 상태
    assert list(obs["ego"][4:7]) == [1.0, 0.0, 0.0]         # 지시등 끔
    for _ in range(40):
        state, info = w.step(0.2, 1.5, rs.TS_RIGHT)
    obs = build_observation(w, state, info, (0.2 / math.radians(35), 0.75, rs.TS_RIGHT))
    cfg = ObsConfig()
    assert obs["ego"][0] == pytest.approx(w.ego.v / cfg.v_max, abs=1e-3)
    assert obs["ego"][1] == pytest.approx(0.2 / math.radians(35), abs=1e-3)
    assert list(obs["ego"][4:7]) == [0.0, 0.0, 1.0]         # 우측 지시등
    assert obs["ego"][3] != 0.0                             # 요레이트


def test_경로는_자차_기준이고_앞을_본다():
    w = h_world()
    state = w.reset()
    for _ in range(60):
        state, info = w.step(0.0, 1.0, 0)
    obs = build_observation(w, state, info, (0.0, 0.5, 0))
    cfg = ObsConfig()
    pts = obs["route"] * cfg.route_ahead
    assert np.all(np.diff(pts[:, 0]) > -1.0)                # 앞으로 간다
    assert abs(pts[0, 1]) < 3.0                             # 바로 앞은 옆으로 많이 안 떨어져 있다
    assert abs(obs["nav"][0]) < 1.0                         # 경로에 붙어 있다
    assert 0.0 < obs["nav"][2] <= 1.0                       # 남은 거리


def test_신호와_물체():
    b = slice_board(load_board(H), 0.0, 250.0, "H_red", signals="always_red")
    x, y, h = b.route.point_at(20.0)  # 60 스텝 뒤 자차는 s≈4m — 세계 물체 범위(80m) 안에 두려고 150→20
    b.scenario.actors = [rs.Actor(id=7, type="vehicle", size=[4.5, 1.8, 1.5],
                                  spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y]})]
    w = World(b)
    state = w.reset()
    for _ in range(60):
        state, info = w.step(0.0, 1.0, 0)
    obs = build_observation(w, state, info, (0.0, 0.5, 0))
    assert obs["signal"][rs.TL_RED] == 1.0
    assert 0.0 < obs["signal"][7] < 1.0                     # 정지선까지 거리
    assert obs["object_mask"][0] == 1.0 and obs["object_mask"][-1] == 0.0
    assert obs["objects"][0, 0] > 0.0                       # 앞에 있다
    assert list(obs["objects"][0, 9:12]) == [1.0, 0.0, 0.0]  # 차량
    assert np.all(obs["objects"][1:] == 0.0)                # 빈 자리는 0


def test_물체는_가까운_순_16개까지():
    b = slice_board(load_board(H), 0.0, 250.0, "H_obj")
    acts = []
    for k in range(20):
        x, y, _h = b.route.point_at(20.0 + 3.0 * k)     # 전부 세계의 물체 범위(80 m) 안
        acts.append(rs.Actor(id=100 + k, type="obstacle", size=[0.5, 0.5, 0.8],
                             spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y + 6.0]}))
    b.scenario.actors = acts
    w = World(b)
    state = w.reset()
    obs = build_observation(w, state, None, (0.0, 0.0, 0))
    assert obs["object_mask"].sum() == 16
    d = np.hypot(obs["objects"][:, 0], obs["objects"][:, 1])
    assert np.all(np.diff(d) >= -1e-6)                      # 가까운 순
