"""관측 사전 -> 신경망 입력.

물체만 따로 둔다(개수 고정 + 마스크). 나머지 묶음은 한 줄로 이어 붙인다 —
2부에서 영상 인코더로 갈아 끼울 때 바꾸는 곳이 여기 하나가 되도록.
"""
import numpy as np
import torch

from vtd_rl.env.observation import ObsConfig

VEC_KEYS = ("ego", "route", "nav", "plan", "signal")
_CFG = ObsConfig()
OBJ_N, OBJ_DIM = _CFG.objects, 12
VEC_DIM = 9 + _CFG.route_points * 2 + 3 + 10 + 11


def flatten_obs(obs: dict):
    vec = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in VEC_KEYS])
    return (vec,
            np.asarray(obs["objects"], dtype=np.float32),
            np.asarray(obs["object_mask"], dtype=np.float32))


def stack_obs(batch):
    parts = [flatten_obs(o) for o in batch]
    return (np.stack([p[0] for p in parts]),
            np.stack([p[1] for p in parts]),
            np.stack([p[2] for p in parts]))


def to_tensors(vec, objs, mask, device):
    tv = torch.as_tensor(vec, dtype=torch.float32, device=device)
    to_ = torch.as_tensor(objs, dtype=torch.float32, device=device)
    tm = torch.as_tensor(mask, dtype=torch.float32, device=device)
    if tv.ndim == 1:
        tv, to_, tm = tv[None], to_[None], tm[None]
    return tv, to_, tm
