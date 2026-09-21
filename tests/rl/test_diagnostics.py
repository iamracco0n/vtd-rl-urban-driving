import numpy as np
import torch

from vtd_rl.rl.diagnostics import (OutcomeCounter, ReturnTracker, policy_drift,
                                   snapshot_policy)


def test_끝난_판의_리턴만_담는다():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 2.0]), np.array([False, False]))
    t.add(np.array([3.0, 4.0]), np.array([True, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 1
    assert abs(s["rollout_return_mean"] - 4.0) < 1e-9      # 0번 환경: 1+3
    assert abs(s["rollout_len_mean"] - 2.0) < 1e-9


def test_끝난_판이_없으면_평균은_None():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 1.0]), np.array([False, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 0 and s["rollout_return_mean"] is None
    assert s["rollout_len_mean"] is None


def test_판이_끝나면_누적이_0에서_다시_시작한다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([2.0]), np.array([True]))
    s = t.stats()
    assert s["rollout_return_n"] == 2
    assert abs(s["rollout_return_mean"] - 3.5) < 1e-9      # (5 + 2) / 2


def test_reset_은_담은_것만_비우고_진행중_누적은_남긴다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([1.0]), np.array([False]))              # 진행 중 누적 1.0
    t.reset()
    assert t.stats()["rollout_return_n"] == 0
    t.add(np.array([2.0]), np.array([True]))
    assert abs(t.stats()["rollout_return_mean"] - 3.0) < 1e-9   # 1 + 2 — 경계를 넘어 이어졌다


def test_종료_사유를_센다():
    c = OutcomeCounter()
    c.add({"outcome": np.array(["goal", "collision", "running"], dtype=object)})
    c.add({"outcome": np.array(["goal", "stalled", "running"], dtype=object)})
    s = c.stats()
    assert s["outcome_goal"] == 2 and s["outcome_collision"] == 1 and s["outcome_stalled"] == 1
    assert "outcome_running" not in s                      # 진행 중은 종료 사유가 아니다


def test_outcome_키가_없으면_아무것도_안_센다():
    c = OutcomeCounter()
    c.add({})
    assert c.stats() == {}


def test_드리프트는_움직인_만큼_커진다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    assert policy_drift(net, ref)["drift_l2"] == 0.0
    with torch.no_grad():
        net.policy.log_std.add_(1.0)
    d = policy_drift(net, ref)
    assert abs(d["drift_l2"] - 2.0 ** 0.5) < 1e-5          # 두 차원에 각각 1.0
    assert d["drift_rel"] > 0.0


def test_스냅샷은_이후_변경에_영향받지_않는다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    with torch.no_grad():
        net.policy.log_std.add_(3.0)
    assert policy_drift(net, ref)["drift_l2"] > 1.0        # clone 이 아니면 0 이 나온다
