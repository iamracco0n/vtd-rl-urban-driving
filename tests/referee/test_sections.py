import os
import random

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.referee.sections import FastSections
from vtd_rl.world.board import load_curriculum

STAGE1 = os.path.join(os.path.dirname(__file__), "..", "..", "curricula", "stage1.json")
_, BOARDS = load_curriculum(STAGE1)


@pytest.mark.parametrize("board", BOARDS, ids=lambda b: b.name)
def test_가장_가까운_점은_원본과_같다(board):
    route = [list(p) for p in board.route.pts]
    slow, fast = rs.score_fma.Sections(route, 5), FastSections(route, 5)
    rng = random.Random(0)
    xs, ys = [p[0] for p in route], [p[1] for p in route]
    queries = [(rng.uniform(min(xs) - 40, max(xs) + 40), rng.uniform(min(ys) - 40, max(ys) + 40))
               for _ in range(200)]
    queries += [(p[0] + 0.3, p[1] - 0.2) for p in route[::37]]
    queries += [tuple(p) for p in route[::53]]
    for x, y in queries:
        assert fast.index_of(x, y) == slow.index_of(x, y)
        assert fast.of(x, y) == slow.of(x, y)


def test_거리가_같으면_작은_번호():
    route = [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [20.0, 0.0]]
    assert FastSections(route, 2).index_of(0.0, 0.0) == 0
    assert FastSections(route, 2).index_of(5.0, 0.0) == rs.score_fma.Sections(route, 2).index_of(5.0, 0.0)
