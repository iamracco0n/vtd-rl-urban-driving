import copy

from vtd_rl import rule_stack as rs
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def rows_along(board, v, n, dt=0.05):
    """경로를 따라 속도 v 로 가는 합성 행."""
    out, s = [], 0.0
    for k in range(n):
        x, y, h = board.route.point_at(s)
        out.append(dict(t=round(k * dt, 2), x=x, y=y, h=h, v=v, tl=0, tid=0, reason="", sig=0,
                        clr=99.9, d_ego=0.0, objs="", cap_by=""))
        s += v * dt
    return out


def test_과속은_항목1_중대():
    b = h_slice()
    rows = rows_along(b, v=80 / 3.6, n=200)
    before = copy.deepcopy(rows)
    res = score_episode(rows, b, sections=5, use_map=False)
    assert rows == before
    assert {i for _, i, _ in res.hits} == {1}
    assert {lv for _, _, lv in res.hits} == {"major"}
    assert res.state[0] == {1: "major"}
    assert res.respawns == {}


def test_규정속도는_감점_없음():
    b = h_slice()
    res = score_episode(rows_along(b, v=40 / 3.6, n=300), b, sections=5, use_map=False)
    assert res.hits == []


def test_지도는_복사한_행에_붙이고_없으면_3456_을_건너뛴다(monkeypatch):
    b = h_slice()
    rows = rows_along(b, v=40 / 3.6, n=50)
    seen, called = [], []
    monkeypatch.setattr(rs.check_lanes, "classify", lambda rr, mp: seen.append((rr, mp)))
    monkeypatch.setattr(rs.score_fma, "item_lane_geometry", lambda *a: called.append(a))
    score_episode(rows, b, use_map=True)
    assert len(seen) == 1 and seen[0][0] is not rows and seen[0][0] == rows
    assert seen[0][1] is rs.load_map()
    assert len(called) == 1
    seen.clear()
    called.clear()
    score_episode(rows, b, use_map=False)
    assert seen == [] and called == []


def test_리스폰은_따로_센다():
    b = h_slice()
    rows = rows_along(b, v=10.0, n=100)
    for r in rows[50:]:
        r["x"] += 20.0
    res = score_episode(rows, b, sections=5, use_map=False)
    assert sum(len(v) for v in res.respawns.values()) == 1
    assert all(i != 15 for _, i, _ in res.hits)
