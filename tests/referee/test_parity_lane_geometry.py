import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Referee, default_judges
from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows

sf = rs.score_fma


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_차로_기하_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(3, 4, 5, 6))
    assert p.ok, p.diff()


def test_지도_없는_심판에는_넣지_않는다():
    assert LaneGeometryJudge in default_judges(True)
    assert LaneGeometryJudge not in default_judges(False)


def _emit_times(name):
    """(행 t, Hit) — step() 이 낸 것과 finish() 가 낸 것을 나눠 돌려준다."""
    sc, rows = episode_rows(name)
    ref = Referee(sc.board, use_map=True, judges=[LaneGeometryJudge])
    stepped = []
    for r in rows:
        stepped += [(r["t"], h) for h in ref.step(dict(r))]
    return stepped, ref.finish()


@pytest.mark.parametrize("name, item, min_s", [("centerline_crosser", 4, sf.CENTER_S),
                                               ("sidewalk_rider", 5, sf.WALK_S)])
def test_중앙선_보도_침범은_최소_시간에_닿는_프레임에_낸다(name, item, min_s):
    stepped, finished = _emit_times(name)
    calls = [(t, h) for t, h in stepped if h.item == item]
    assert calls and not [h for h in finished if h.item == item]
    for t, h in calls:
        assert min_s - 1e-6 <= t - h.t <= min_s + 0.1, (t, h)


def test_차로_유지_판정은_끝을_기다리지_않고_2_5초_뒤에_낸다():
    stepped, finished = _emit_times("centerline_crosser")
    calls = [(t, h) for t, h in stepped if h.item == 3]
    assert calls and not [h for h in finished if h.item == 3]
    for t, h in calls:
        assert t >= h.t + 2.5, (t, h)          # 2.5: item_lane_geometry 안의 LC_WIN
