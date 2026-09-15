import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world.board import load_board
from vtd_rl.world.route import RouteIndex
from vtd_rl.world.signals import (REPORT_RANGE, RouteSignal, SignalProgram, SignalReporter,
                                  programs_for, route_signals)

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def test_코스_H_같은_방향_신호만():
    b = load_board(H)
    db = rs.map_db()
    sig = route_signals(b.route, db["tl_map"], db["stoplines_all"])
    assert [s.tl_id for s in sig[:3]] == [151, 155, 161]     # 반대 방향 150·152·157 은 빠진다
    assert sig[0].s == pytest.approx(96.0, abs=3.0)
    assert all(a.s < c.s for a, c in zip(sig, sig[1:]))


def test_신호_주기():
    p = SignalProgram("cycle", green=10.0, yellow=2.0, red=8.0, offset=0.0)
    assert p.state(0.0) == rs.TL_GREEN
    assert p.state(10.5) == rs.TL_YELLOW
    assert p.state(12.5) == rs.TL_RED
    assert p.state(20.5) == rs.TL_GREEN
    assert SignalProgram().state(999.0) == rs.TL_GREEN


def test_프로그램_묶음():
    sig = [RouteSignal(10.0, 3), RouteSignal(50.0, 9)]
    g = programs_for("always_green", sig, random.Random(0))
    assert set(g) == {3, 9} and g[3].mode == "always_green"
    c1 = programs_for("cycle", sig, random.Random(1))
    c2 = programs_for("cycle", sig, random.Random(1))
    assert c1[3].offset == c2[3].offset                      # 같은 시드 = 같은 위상
    with pytest.raises(ValueError):
        programs_for("blink", sig, random.Random(0))


def test_알림_규칙():
    sig = [RouteSignal(100.0, 7)]
    lane = [{"j": 0}] * 150 + [{"j": 1}] * 10 + [None] * 40
    rep = SignalReporter(sig, lane, {7: SignalProgram()})
    assert rep.report(100.0 - REPORT_RANGE - 1.0, 0, 0.0) == (-1, rs.TL_UNSET)   # 너무 멀다
    assert rep.report(20.0, 20, 0.0) == (7, rs.TL_GREEN)
    assert rep.report(97.0, 97, 0.0) == (7, rs.TL_GREEN)                         # 정지선 조금 지남
    assert rep.report(155.0, 155, 0.0) == (-1, rs.TL_UNSET)                      # 교차로 안
    assert rep.report(120.0, 120, 0.0) == (-1, rs.TL_UNSET)                      # 완전히 지남
    assert rep.report(20.0, 190, 0.0) == (7, rs.TL_GREEN)                        # 차로계획 칸이 None


def test_짧은_직선에는_신호가_없다():
    r = RouteIndex([(float(i), 0.0) for i in range(50)])
    assert route_signals(r, {1: [1000.0, 1000.0]}, [[1000.0, 1000.0, 0.0]]) == []
