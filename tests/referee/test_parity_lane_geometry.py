import pytest

from vtd_rl.referee.core import Referee, default_judges
from vtd_rl.referee.judges.lane_geometry import LaneGeometryJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_차로_기하_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(3, 4, 5, 6))
    assert p.ok, p.diff()


def test_지도_없는_심판에는_넣지_않는다():
    assert LaneGeometryJudge in default_judges(True)
    assert LaneGeometryJudge not in default_judges(False)


def test_차로_유지_판정은_2_5초_뒤에_낸다():
    sc, rows = episode_rows("centerline_crosser")
    ref = Referee(sc.board, use_map=True, judges=[LaneGeometryJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            if h.item == 3:
                assert r["t"] >= h.t + 2.5
    ref.finish()
    assert any(h.item == 4 for h in ref.hits)
