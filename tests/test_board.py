import os

import pytest

from vtd_rl.world.board import Board, load_board, load_curriculum, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage1.json")


def test_코스_H_불러오기():
    b = load_board(H)
    assert len(b.route.pts) == len(b.lane_plan) == 2015     # 원본 ego_route 와 차로계획이 짝
    assert b.route.total == pytest.approx(2807, abs=2)
    x, y, h = b.start_pose
    assert (x, y) == pytest.approx(b.route.pts[0])
    assert h == pytest.approx(b.route.heading_at(0))
    assert b.signals == "always_green"


def test_짝이_안_맞으면_거부():
    b = load_board(H)
    with pytest.raises(ValueError):
        Board("bad", b.scenario, b.lane_plan[:-1])


def test_자르기():
    b = load_board(H)
    s = slice_board(b, 0.0, 250.0, "H_0_250")
    assert s.name == "H_0_250"
    assert 245.0 <= s.route.total <= 251.0
    assert len(s.route.pts) == len(s.lane_plan)
    assert s.scenario.actors == [] and s.scenario.lights == []
    assert s.goal == pytest.approx(s.route.pts[-1])
    assert s.scenario.duration >= 60.0


def test_단계1_목록():
    name, boards = load_curriculum(STAGE1)
    assert name.startswith("빈 경로")
    assert [b.name for b in boards] == ["course_A", "course_B", "course_D", "course_E", "course_G", "course_H"]
    assert all(b.signals == "always_green" for b in boards)
    assert all(len(b.route.pts) == len(b.lane_plan) for b in boards)
