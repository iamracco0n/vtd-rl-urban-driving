import numpy as np
import torch

from vtd_rl.env.action import TURNS, action_space
from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import flatten_obs, stack_obs, to_tensors
from vtd_rl.policy.net import DrivePolicy, PolicyConfig


def sample_obs(seed=0):
    space = observation_space(ObsConfig())
    space.seed(seed)
    return space.sample()


def test_출력_크기():
    net = DrivePolicy()
    vec, objs, mask = to_tensors(*stack_obs([sample_obs(i) for i in range(5)]), torch.device("cpu"))
    mean, log_std, logits = net(vec, objs, mask)
    assert mean.shape == (5, 2) and logits.shape == (5, len(TURNS))
    assert log_std.shape == (2,)


def test_행동은_행동공간_안에_있다():
    net, space = DrivePolicy(), action_space()
    for seed in range(5):
        a = net.act(sample_obs(seed))
        assert space.contains(a), a
    g = torch.Generator().manual_seed(0)
    sampled = net.act(sample_obs(0), deterministic=False, generator=g)
    assert space.contains(sampled)


def test_마스크된_물체는_결과를_바꾸지_않는다():
    net = DrivePolicy()
    obs = sample_obs(1)
    obs["object_mask"][:] = 0.0
    obs["objects"][:] = 0.0
    a = net.act(obs)
    obs2 = dict(obs)
    obs2["objects"] = obs["objects"].copy()
    obs2["objects"][3] = 0.9                      # 마스크가 0 인 자리를 채워도
    b = net.act(obs2)
    assert np.allclose(a["control"], b["control"], atol=1e-6) and a["turn"] == b["turn"]


def test_물체_순서를_바꿔도_결과가_같다():
    net = DrivePolicy()
    obs = sample_obs(2)
    obs["object_mask"][:] = 0.0
    obs["object_mask"][:3] = 1.0
    a = net.act(obs)
    obs2 = dict(obs)
    obs2["objects"] = obs["objects"].copy()
    obs2["objects"][[0, 2]] = obs2["objects"][[2, 0]]
    b = net.act(obs2)
    assert np.allclose(a["control"], b["control"], atol=1e-5) and a["turn"] == b["turn"]


def test_저장하고_불러오면_같은_행동(tmp_path):
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    path = tmp_path / "policy.pt"
    net.save(str(path))
    other = DrivePolicy.load(str(path))
    obs = sample_obs(4)
    a, b = net.act(obs), other.act(obs)
    assert np.allclose(a["control"], b["control"], atol=1e-6) and a["turn"] == b["turn"]
    assert other.cfg.trunk == (32, 32)
