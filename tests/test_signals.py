import os
import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.teacher.shadow import ShadowTeacher
from vtd_rl.world.board import load_board, load_curriculum
from vtd_rl.world.route import RouteIndex
from vtd_rl.world.signals import (REPORT_RANGE, RouteSignal, SignalProgram, SignalReporter,
                                  programs_for, route_signals)

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage1.json")


def test_코스_H_같은_방향_신호만():
    b = load_board(H)
    db = rs.map_db()
    sig = route_signals(b.route, db["tl_map"], db["stoplines_all"])
    assert [s.tl_id for s in sig[:3]] == [151, 155, 161]     # 반대 방향 150·152·157 은 빠진다
    assert sig[0].s == pytest.approx(96.0, abs=3.0)
    assert all(a.s < c.s for a, c in zip(sig, sig[1:]))
    assert 221 in [s.tl_id for s in sig]                     # 근처에 도색선이 없는 신호 — DB 점 투영으로 남는다


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
    # 신호가 없으면 VTD 처럼 tl_id 0 (규칙 스택 behavior 의 우회전 녹색 래치가 0 을 '신호 없음'으로 읽는다)
    assert rep.report(100.0 - REPORT_RANGE - 1.0, 0, 0.0) == (0, rs.TL_UNSET)    # 너무 멀다
    assert rep.report(20.0, 20, 0.0) == (7, rs.TL_GREEN)
    assert rep.report(97.0, 97, 0.0) == (7, rs.TL_GREEN)                         # 정지선 3 m 앞
    assert rep.report(103.0, 103, 0.0) == (7, rs.TL_GREEN)                       # 정지선 3 m 지남(PASSED_MARGIN 안)
    assert rep.report(155.0, 155, 0.0) == (0, rs.TL_UNSET)                       # 교차로 안
    assert rep.report(120.0, 120, 0.0) == (0, rs.TL_UNSET)                       # 완전히 지남
    assert rep.report(20.0, 190, 0.0) == (7, rs.TL_GREEN)                        # 차로계획 칸이 None


def test_짧은_직선에는_신호가_없다():
    r = RouteIndex([(float(i), 0.0) for i in range(50)])
    assert route_signals(r, {1: [1000.0, 1000.0]}, [[1000.0, 1000.0, 0.0]]) == []


def _loop_twice():
    """동쪽 y=0(0→100) → 북 → 서쪽 y=40 → 남 → 다시 동쪽 y=0. 같은 교차로를 같은 방향으로 두 번 지난다."""
    legs = [((0, 0), (100, 0)), ((100, 0), (100, 40)), ((100, 40), (0, 40)), ((0, 40), (0, 0)),
            ((0, 0), (100, 0))]
    pts = [(0.0, 0.0)]
    for (x0, y0), (x1, y1) in legs:
        n = int(max(abs(x1 - x0), abs(y1 - y0)))
        pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return RouteIndex(pts)


def test_도색선은_경로의_모든_통과에_붙는다():
    r = _loop_twice()
    tl = {8: [50.5, -5.5], 7: [50.5, -5.5],      # 한 자리 두 id — 작은 id 로
          3: [50.0, 43.0]}                        # 서쪽 통과 옆 신호, 그 도색선은 동향이라 내 방향이 아니다
    sl = [[50.4, 0.0, 0.0], [50.6, -3.0, 0.0],   # 차로별 도색선 둘(s 50.4·50.6 — round(s) 로는 갈린다)
          [50.0, 40.5, 0.0]]
    sig = route_signals(r, tl, sl)
    assert [s.tl_id for s in sig] == [7, 7]
    assert sig[0].s == pytest.approx(50.4, abs=0.3)
    assert sig[1].s == pytest.approx(50.4 + 280.0, abs=0.3)


def test_다시_지나는_교차로의_반대_방향_통과에_끌려가지_않는다():
    # 신호 DB 점은 나중의 반대 방향 통과(y=8)에 더 가깝다 — 가장 가까운 한 곳만 보면 방위에서 버려진다
    pts = ([(float(x), 0.0) for x in range(101)] + [(100.0, float(y)) for y in range(1, 9)]
           + [(float(x), 8.0) for x in range(99, -1, -1)])
    r = RouteIndex(pts)
    sig = route_signals(r, {5: [50.0, 5.5]}, [[50.0, 0.5, 0.0]])
    assert [s.tl_id for s in sig] == [5]
    assert sig[0].s == pytest.approx(50.0, abs=0.3)


def test_도색선이_없는_신호는_DB_점을_투영한다():
    r = RouteIndex([(float(i), 0.0) for i in range(201)])
    sig = route_signals(r, {9: [150.0, 1.0]}, [[130.0, 0.0, 0.0]])      # 가장 가까운 도색선이 20 m — 짝이 없다
    assert sig == [RouteSignal(pytest.approx(150.0), 9)]


def test_알림은_s_순서로_정렬한다():
    lane = [{"j": 0}] * 400
    rep = SignalReporter([RouteSignal(300.0, 9), RouteSignal(100.0, 7)], lane,
                         {7: SignalProgram(), 9: SignalProgram()})
    assert rep.report(20.0, 20, 0.0) == (7, rs.TL_GREEN)
    assert rep.report(200.0, 200, 0.0) == (9, rs.TL_GREEN)


@pytest.mark.parametrize("board", load_curriculum(STAGE1)[1], ids=lambda b: b.name)
def test_선생님이_아는_경로_위_신호_정지선을_world_도_안다(board):
    db = rs.map_db()
    sig = route_signals(board.route, db["tl_map"], db["stoplines_all"])
    lines = ShadowTeacher(board).stack._tl_lines           # [(s, x, y)] — 규칙 스택이 만든 경로 위 신호 정지선
    assert lines
    missing = [round(s_q) for s_q, _x, _y in lines if not any(abs(r.s - s_q) <= 10.0 for r in sig)]
    assert missing == []


def test_시험용_신호_운용():
    sig = [RouteSignal(10.0, 3)]
    red = programs_for("always_red", sig, random.Random(0))
    flash = programs_for("always_flash", sig, random.Random(0))
    assert red[3].state(123.0) == rs.TL_RED
    assert flash[3].state(0.0) == 6
