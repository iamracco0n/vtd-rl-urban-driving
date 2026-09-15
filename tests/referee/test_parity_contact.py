import pytest

from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_접촉과_횡단보도_정지_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(11, 12, 14))
    assert p.ok, p.diff()
