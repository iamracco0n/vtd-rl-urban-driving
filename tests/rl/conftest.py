import pytest
import torch

from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import stack_obs, to_tensors


@pytest.fixture
def obs_batch():
    def _make(n=4, seed=0):
        space = observation_space(ObsConfig())
        space.seed(seed)
        return to_tensors(*stack_obs([space.sample() for _ in range(n)]), torch.device("cpu"))
    return _make
