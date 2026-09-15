import pytest

from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_지시등_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(13,))
    assert p.ok, p.diff()
