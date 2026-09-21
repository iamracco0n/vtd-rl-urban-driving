import torch

from vtd_rl.env.action import action_space
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.evaluate import EpisodeOutcome, _summary
from vtd_rl.policy.net import DrivePolicy, PolicyConfig


def test_log_std_는_기울기를_잃지_않는다(obs_batch):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    vec, objs, mask = obs_batch()
    _mean, log_std, _logits = net(vec, objs, mask)
    log_std.sum().backward()
    assert net.log_std.grad is not None and torch.all(net.log_std.grad != 0.0)


def test_한_걸음_뒤에_잘라_범위를_지킨다(obs_batch):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_min=-2.0, log_std_max=0.5))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([-5.0, 3.0]))
    net.clamp_log_std()
    assert torch.allclose(net.log_std.detach(), torch.tensor([-2.0, 0.5]))
    # 바닥에 있어도 다음 기울기는 살아 있다
    _m, log_std, _l = net(*obs_batch())
    log_std.sum().backward()
    assert torch.all(net.log_std.grad != 0.0)


def test_표본은_자르기_전_로그확률을_준다(obs_batch):
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32), log_std_max=2.0))
    with torch.no_grad():
        net.log_std.copy_(torch.tensor([1.5, 1.5]))     # σ≈4.5 — 표본이 확실히 잘린다
    vec, objs, mask = obs_batch(6)
    out = net.sample(vec, objs, mask, generator=torch.Generator().manual_seed(1))
    assert out["raw"].shape == (6, 2) and out["control"].shape == (6, 2)
    assert torch.all(out["control"] <= 1.0) and torch.all(out["control"] >= -1.0)
    assert out["turn"].shape == (6,) and out["log_prob"].shape == (6,)
    assert not torch.allclose(out["raw"], out["control"])      # 실제로 잘린 표본이 있다
    again, entropy = net.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    assert torch.allclose(again, out["log_prob"], atol=1e-5)
    assert torch.all(entropy > 0.0)
    # 잘린 값이 아니라 원표본으로 계산한다
    clipped_lp, _ = net.evaluate_actions(vec, objs, mask, out["control"], out["turn"])
    assert not torch.allclose(clipped_lp, out["log_prob"], atol=1e-6)


def test_행동은_환경_공간_안이다():
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    space = observation_space(ObsConfig())
    space.seed(3)
    obs = space.sample()
    a = net.act(obs, deterministic=False, generator=torch.Generator().manual_seed(0))
    assert action_space().contains(a)


def test_완주하지_못한_판은_0점():
    eps = [EpisodeOutcome("A", 0, "goal", 100, 1.0, 98.0, []),
           EpisodeOutcome("B", 0, "stalled", 600, -5.0, 99.0, []),
           EpisodeOutcome("D", 0, "offroad", 90, -50.0, 100.0, [])]
    s = _summary(eps)
    assert s["goal_rate"] == 1 / 3
    assert s["mean_score"] == 98.0 / 3                       # 완주한 판만 점수를 남긴다
    assert s["mean_score_raw"] == (98.0 + 99.0 + 100.0) / 3
