import numpy as np
import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard, load_dir, load_shard, save_shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM


def fake_shard(n=7, seed=0, name="course_H"):
    rng = np.random.default_rng(seed)
    return Shard(vec=rng.random((n, VEC_DIM), dtype=np.float32),
                 objs=rng.random((n, OBJ_N, OBJ_DIM), dtype=np.float32),
                 mask=(rng.random((n, OBJ_N)) > 0.5).astype(np.float32),
                 control=rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                 turn=rng.integers(0, 3, n).astype(np.int64),
                 meta={"board": name, "seed": seed, "beta": 1.0, "outcome": "goal"})


def test_저장하고_읽으면_같다(tmp_path):
    s = fake_shard()
    path = tmp_path / "a.npz"
    save_shard(s, str(path))
    r = load_shard(str(path))
    for a, b in ((s.vec, r.vec), (s.objs, r.objs), (s.mask, r.mask), (s.control, r.control)):
        assert np.array_equal(a, b)
    assert np.array_equal(s.turn, r.turn) and r.turn.dtype == np.int64
    assert r.meta == s.meta


def test_데이터셋_모으기와_배치():
    ds = DaggerDataset()
    ds.add(fake_shard(5, 1))
    ds.add(fake_shard(9, 2))
    assert len(ds) == 14 and ds.round_counts == [5, 9]
    vec, objs, mask, control, turn = ds.arrays()
    assert vec.shape == (14, VEC_DIM) and turn.shape == (14,)
    g = torch.Generator().manual_seed(0)
    seen, batches = 0, 0
    for batch in ds.batches(4, generator=g):
        assert batch[0].shape[0] <= 4 and isinstance(batch[0], torch.Tensor)
        seen += batch[0].shape[0]
        batches += 1
    assert seen == 14 and batches == 4


def test_폴더에서_읽기(tmp_path):
    save_shard(fake_shard(3, 1), str(tmp_path / "r0-course_A.npz"))
    save_shard(fake_shard(4, 2), str(tmp_path / "r0-course_B.npz"))
    ds = load_dir(str(tmp_path))
    assert len(ds) == 7 and len(ds.round_counts) == 2
