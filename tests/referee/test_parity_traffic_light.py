import pytest

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_신호_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(7, 8, 9))
    assert p.ok, p.diff()


def test_적색_무정차는_선을_넘는_프레임에_낸다():
    sc, rows = episode_rows("red_runner")
    ref = Referee(sc.board, use_map=False, judges=[TrafficLightJudge])
    emitted_at = None
    for r in rows:
        if ref.step(dict(r)) and emitted_at is None:
            emitted_at = r["t"]
    ref.finish()
    assert emitted_at is not None
    assert [h.t for h in ref.hits if h.item == 7] == [emitted_at]


def test_녹색_정차는_정차가_끝날_때_낸다():
    sc, rows = episode_rows("green_idler")
    ref = Referee(sc.board, use_map=False, judges=[TrafficLightJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            assert h.item == 8 and r["t"] - h.t >= 10.0
    ref.finish()
    assert [h.level for h in ref.hits] == ["major"]
