import pytest

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.pedestrian import PedestrianJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_보행자_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(10,))
    assert p.ok, p.diff()


def test_판정은_1초_뒤에_낸다():
    sc, rows = episode_rows("ped_ignorer")
    ref = Referee(sc.board, use_map=False, judges=[PedestrianJudge])
    for r in rows:
        for h in ref.step(dict(r)):
            assert r["t"] > h.t + 1.0
    ref.finish()
    assert any(h.item == 10 for h in ref.hits)
