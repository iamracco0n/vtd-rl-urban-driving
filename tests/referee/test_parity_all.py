import os

import pytest

from vtd_rl.drivers.scripted import TeacherDriver
from vtd_rl.referee.parity import compare
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows
from vtd_rl.rollout import run_episode
from vtd_rl.world.board import load_curriculum

STAGE1 = os.path.join(os.path.dirname(__file__), "..", "..", "curricula", "stage1.json")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_대본_판_전_항목_일치(name):
    sc, rows = episode_rows(name)
    p = compare(sc.board, rows)
    assert p.ok, p.diff()


@pytest.mark.slow
@pytest.mark.parametrize("board", load_curriculum(STAGE1)[1], ids=lambda b: b.name)
def test_선생님_연습_코스_전_항목_일치(board):
    run = run_episode(board, TeacherDriver(board))
    p = compare(board, run.rows)
    assert p.ok, p.diff()
