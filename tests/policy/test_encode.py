import numpy as np
import torch

from vtd_rl.env.observation import ObsConfig, observation_space
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM, flatten_obs, stack_obs, to_tensors


def sample_obs(seed=0):
    space = observation_space(ObsConfig())
    space.seed(seed)
    return space.sample()


def test_평탄화_크기와_형():
    vec, objs, mask = flatten_obs(sample_obs())
    assert vec.shape == (VEC_DIM,) and vec.dtype == np.float32
    assert objs.shape == (OBJ_N, OBJ_DIM) and mask.shape == (OBJ_N,)
    assert VEC_DIM == 9 + 20 * 2 + 3 + 10 + 11


def test_평탄화는_값을_그대로_옮긴다():
    obs = sample_obs(3)
    vec, objs, mask = flatten_obs(obs)
    assert np.array_equal(vec[:9], obs["ego"])
    assert np.array_equal(vec[9:49], obs["route"].reshape(-1))
    assert np.array_equal(vec[49:52], obs["nav"])
    assert np.array_equal(vec[52:62], obs["plan"])
    assert np.array_equal(vec[62:73], obs["signal"])
    assert np.array_equal(objs, obs["objects"]) and np.array_equal(mask, obs["object_mask"])


def test_묶기와_텐서():
    batch = [sample_obs(i) for i in range(4)]
    vec, objs, mask = stack_obs(batch)
    assert vec.shape == (4, VEC_DIM) and objs.shape == (4, OBJ_N, OBJ_DIM)
    tv, to_, tm = to_tensors(vec, objs, mask, torch.device("cpu"))
    assert tv.shape == (4, VEC_DIM) and to_.dtype == torch.float32
    one = to_tensors(*flatten_obs(batch[0]), torch.device("cpu"))
    assert one[0].shape == (1, VEC_DIM) and one[1].shape == (1, OBJ_N, OBJ_DIM)
