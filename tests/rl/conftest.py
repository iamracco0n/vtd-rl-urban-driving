import pytest
import torch

from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import stack_obs, to_tensors
from vtd_rl.policy.net import PolicyConfig
from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig


@pytest.fixture
def obs_batch():
    def _make(n=4, seed=0):
        space = observation_space(ObsConfig())
        space.seed(seed)
        return to_tensors(*stack_obs([space.sample() for _ in range(n)]), torch.device("cpu"))
    return _make


@pytest.fixture
def small_ac():
    """테스트용 작은 행위자-비평가 — 실제 크기 대신 빠르게 도는 크기로."""
    def _make():
        cfg = ActorCriticConfig(policy=PolicyConfig(obj_hidden=16, obj_out=16, trunk=(32, 32)),
                                value_hidden=(32, 32), obj_hidden=16, obj_out=16)
        return ActorCritic(cfg)
    return _make
