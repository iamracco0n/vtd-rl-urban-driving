import pytest

from vtd_rl.referee.oracle import score_episode
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_대본_판은_채점기에서_기대한_항목을_일으킨다(name):
    sc, rows = episode_rows(name)
    fired = {i for _, i, _ in score_episode(rows, sc.board, sections=5, use_map=True).hits}
    assert sc.expect <= fired, f"{name}: 기대 {sorted(sc.expect)} / 채점기 {sorted(fired)}"
    assert not (sc.forbid & fired), f"{name}: 금지 {sorted(sc.forbid)} / 채점기 {sorted(fired)}"
