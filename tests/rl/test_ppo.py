import torch

from vtd_rl.policy.dataset import DaggerDataset, Shard
from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
from vtd_rl.rl.buffer import RolloutBuffer
from vtd_rl.rl.ppo import PPOConfig, dagger_batches, imitation_coef, ppo_losses, update
import numpy as np


def filled_buffer(net, n_steps=4, n_envs=2):
    buf = RolloutBuffer(n_steps, n_envs, torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    for t in range(n_steps):
        vec = torch.randn(n_envs, VEC_DIM)
        objs = torch.randn(n_envs, OBJ_N, OBJ_DIM)
        mask = torch.ones(n_envs, OBJ_N)
        out = net.act(vec, objs, mask, generator=g)
        buf.add(vec=vec, objs=objs, mask=mask, raw=out["raw"], turn=out["turn"],
                log_prob=out["log_prob"], value=out["value"],
                reward=torch.randn(n_envs), done=torch.zeros(n_envs),
                valid=torch.ones(n_envs))
    buf.compute_gae(last_value=torch.zeros(n_envs))
    return buf


def toy_dagger(n=64):
    rng = np.random.default_rng(0)
    ds = DaggerDataset()
    ds.add(Shard(rng.standard_normal((n, VEC_DIM), dtype=np.float32),
                 np.zeros((n, OBJ_N, OBJ_DIM), np.float32), np.zeros((n, OBJ_N), np.float32),
                 rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                 rng.integers(0, 3, n).astype(np.int64), {"board": "toy"}))
    return ds


def test_처음_업데이트에서는_비율이_1이라_잘림이_없다(small_ac):
    net = small_ac()
    buf = filled_buffer(net)
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    loss, parts = ppo_losses(net, batch, PPOConfig())
    assert abs(parts["approx_kl"]) < 1e-6 and parts["clip_frac"] == 0.0
    assert torch.isfinite(loss)


def test_모방_배치는_끝없이_돈다():
    it = dagger_batches(toy_dagger(8), 4, generator=torch.Generator().manual_seed(0))
    assert sum(1 for _ in zip(range(10), it)) == 10   # 두 배치짜리를 열 번 꺼내도 안 끊긴다


def test_모방_계수는_반감기로_준다():
    cfg = PPOConfig(imitation_coef0=1.0, imitation_half_life=100)
    assert imitation_coef(0, cfg) == 1.0
    assert abs(imitation_coef(100, cfg) - 0.5) < 1e-9
    assert abs(imitation_coef(200, cfg) - 0.25) < 1e-9


def test_업데이트가_돌고_범위를_지킨다(small_ac):
    net = small_ac()
    with torch.no_grad():
        net.policy.log_std.copy_(torch.tensor([-1.9, 0.45]))
    buf = filled_buffer(net, n_steps=8, n_envs=4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    it = dagger_batches(toy_dagger(), 32, generator=torch.Generator().manual_seed(0))
    out = update(net, opt, buf, it, PPOConfig(epochs=2, minibatch=8), step=0,
                 generator=torch.Generator().manual_seed(0))
    assert set(out) >= {"policy", "value", "entropy", "approx_kl", "clip_frac", "imitation"}
    lo, hi = net.policy.cfg.log_std_min, net.policy.cfg.log_std_max
    assert torch.all(net.policy.log_std >= lo - 1e-6) and torch.all(net.policy.log_std <= hi + 1e-6)


def test_가치_손실이_가치를_목표로_당긴다(small_ac):
    net = small_ac()
    buf = filled_buffer(net, n_steps=8, n_envs=4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    before = None
    for k in range(5):
        batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(k))))
        loss, parts = ppo_losses(net, batch, PPOConfig(entropy_coef=0.0))
        before = parts["value"] if before is None else before
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    batch = next(iter(buf.batches(1000, generator=torch.Generator().manual_seed(0))))
    _loss, parts = ppo_losses(net, batch, PPOConfig(entropy_coef=0.0))
    assert parts["value"] < before
