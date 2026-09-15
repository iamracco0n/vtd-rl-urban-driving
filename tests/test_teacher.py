import os

import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.rollout import run_teacher_episode
from vtd_rl.teacher.shadow import ShadowTeacher
from vtd_rl.world.board import load_board, load_curriculum, slice_board
from vtd_rl.world.world import World

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STAGE1 = os.path.join(os.path.dirname(__file__), "..", "curricula", "stage1.json")


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_선생님은_출발한다():
    b = h_slice()
    w = World(b)
    s = w.reset()
    t = ShadowTeacher(b)
    for _ in range(40):
        cmd = t.act(s, w.clock)
        s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
    assert info.v > 1.0
    t.act(s, w.clock)                     # 방금 받은 State 의 speed 는 선생님이 읽을 때 채워진다
    assert s.speed > 0.5                  # 위치 미분 추정


def test_선생님은_신호_151을_보고_지난다():
    b = h_slice()
    w = World(b)
    s = w.reset()
    t = ShadowTeacher(b)
    seen = set()
    info = None
    while info is None or not info.done:
        cmd = t.act(s, w.clock)
        s, info = w.step(cmd.steer, cmd.accel, cmd.turn)
        seen.add(s.tl_id)
    assert 151 in seen
    assert info.outcome == "goal"


def test_한_판_달리기와_CSV(tmp_path):
    csv = tmp_path / "run.csv"
    r = run_teacher_episode(h_slice(), log_csv=str(csv))
    assert r.outcome == "goal"
    assert 240.0 - 10.0 <= r.distance <= 251.0
    assert r.steps == round(r.sim_time / 0.05)
    assert r.steps_per_sec > 0
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] + "\n" == rs.HEADER
    assert len(lines) - 1 == r.steps


@pytest.mark.slow
@pytest.mark.parametrize("board", load_curriculum(STAGE1)[1], ids=lambda b: b.name)
def test_선생님은_단계1_판을_완주한다(board):
    # 설계 §8.1 — 선생님이 world 에서 단계 ① 판을 완주하지 못하면 world 가 틀린 것이다
    r = run_teacher_episode(board)
    assert r.outcome == "goal", f"{board.name}: {r.outcome} · {r.distance:.0f}/{board.route.total:.0f} m · {r.sim_time:.1f} s"
