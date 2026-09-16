"""판 하나 = 조각(.npz) 하나. DAgger 는 라운드마다 조각을 더해 가며 전부로 학습한다.

조각은 `runs/.../data/` 에만 쌓고 레포에는 넣지 않는다(.gitignore).
"""
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Shard:
    vec: np.ndarray
    objs: np.ndarray
    mask: np.ndarray
    control: np.ndarray
    turn: np.ndarray
    meta: dict = field(default_factory=dict)


def save_shard(shard: Shard, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, vec=shard.vec, objs=shard.objs, mask=shard.mask,
                        control=shard.control, turn=shard.turn,
                        meta=np.array(json.dumps(shard.meta, ensure_ascii=False)))


def load_shard(path: str) -> Shard:
    with np.load(path, allow_pickle=False) as z:
        return Shard(z["vec"], z["objs"], z["mask"], z["control"],
                     z["turn"].astype(np.int64), json.loads(str(z["meta"])))


class DaggerDataset:
    def __init__(self):
        self._shards: list = []

    def add(self, shard: Shard):
        self._shards.append(shard)

    def __len__(self):
        return sum(len(s.turn) for s in self._shards)

    @property
    def round_counts(self):
        return [len(s.turn) for s in self._shards]

    def arrays(self):
        if not self._shards:
            raise ValueError("조각이 없다")
        return (np.concatenate([s.vec for s in self._shards]),
                np.concatenate([s.objs for s in self._shards]),
                np.concatenate([s.mask for s in self._shards]),
                np.concatenate([s.control for s in self._shards]),
                np.concatenate([s.turn for s in self._shards]))

    def batches(self, batch_size: int, generator=None, device=None):
        vec, objs, mask, control, turn = self.arrays()
        order = torch.randperm(len(turn), generator=generator)
        tensors = [torch.as_tensor(a, device=device) for a in (vec, objs, mask, control)]
        tensors.append(torch.as_tensor(turn, dtype=torch.int64, device=device))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yield tuple(t[idx] for t in tensors)


def load_dir(path: str) -> DaggerDataset:
    ds = DaggerDataset()
    for name in sorted(glob.glob(os.path.join(path, "*.npz"))):
        ds.add(load_shard(name))
    return ds
