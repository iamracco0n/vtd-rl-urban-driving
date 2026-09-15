import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.contact import CrosswalkStopJudge
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows

sf = rs.score_fma


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_접촉과_횡단보도_정지_판정_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows, items=(11, 12, 14))
    assert p.ok, p.diff()


def test_횡단보도_위_정지는_CW_STOP_S_에_닿는_프레임에_낸다():
    sc, rows = episode_rows("crosswalk_stopper")
    ref = Referee(sc.board, use_map=False, judges=[CrosswalkStopJudge])
    calls = []
    for r in rows:
        calls += [(r["t"], h) for h in ref.step(dict(r))]
    assert ref.finish() == []
    assert [h.item for _, h in calls] == [12]
    for t, h in calls:
        assert sf.CW_STOP_S - 1e-6 <= t - h.t <= sf.CW_STOP_S + 0.1, (t, h)
