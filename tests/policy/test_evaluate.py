import pytest
import torch

from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, run_policy_episode
from vtd_rl.policy.net import DrivePolicy, PolicyConfig
from vtd_rl.env.drive_env import VtdDriveEnv
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def short_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_학습_전_학생도_판을_끝낸다():
    # 학습 전 신경망 초기값은 무작위다 — 이 시드는 재현성만을 위한 것이고, 테스트 자체는
    # 학습 전 정책이 어떤 결과(goal/offroad/collision/timeout/stalled)에 이르든 통과해야
    # 한다. 세계는 60 초(10 Hz 기준 ~600 스텝) 진행이 없으면 stalled 로 끝나므로, 1000
    # 스텝이면 초기화가 무엇이든 반드시 어떤 결과로든 끝난다(구조적 종료 — 운 좋은 시드에
    # 기대지 않는다. seed 0~5 에서 모두 확인함, task-7-report.md 참고).
    torch.manual_seed(0)
    board = short_board()
    env = VtdDriveEnv([board])
    out = run_policy_episode(env, DrivePolicy(PolicyConfig(trunk=(32, 32))), board.name, 0,
                             max_steps=1000)
    env.close()
    assert out.outcome in ("goal", "offroad", "collision", "timeout", "stalled")
    assert out.steps > 0 and len(out.sheet) == 5


def test_평가_요약():
    torch.manual_seed(0)      # 학습 전 신경망 초기값은 무작위다 — 결과 재현을 위해 고정한다
    boards = [short_board()]
    res = evaluate_policy(DrivePolicy(PolicyConfig(trunk=(32, 32))), boards, seeds=(0, 1))
    assert 0.0 <= res["goal_rate"] <= 1.0 and len(res["episodes"]) == 2
    assert res["mean_score"] <= 100.0


@pytest.mark.slow
def test_선생님_기준():
    res = evaluate_teacher([short_board()], seeds=(0,))
    assert res["goal_rate"] == 1.0 and res["mean_score"] > 80.0
