import torch

from vtd_rl.rl.buffer import RolloutBuffer


def fill(buf, rewards, dones, values, n_envs=1, valids=None):
    valids = valids or [1.0] * len(rewards)
    for r, d, v, ok in zip(rewards, dones, values, valids):
        buf.add(vec=torch.zeros(n_envs, 73), objs=torch.zeros(n_envs, 16, 12),
                mask=torch.zeros(n_envs, 16), raw=torch.zeros(n_envs, 2),
                turn=torch.zeros(n_envs, dtype=torch.long), log_prob=torch.zeros(n_envs),
                value=torch.full((n_envs,), float(v)), reward=torch.full((n_envs,), float(r)),
                done=torch.full((n_envs,), float(d)), valid=torch.full((n_envs,), float(ok)))


def test_GAE_는_손으로_센_값과_같다():
    buf = RolloutBuffer(3, 1, torch.device("cpu"))
    fill(buf, rewards=[1.0, 1.0, 1.0], dones=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0])
    buf.compute_gae(last_value=torch.zeros(1), gamma=1.0, lam=1.0)
    # 감쇠 없음·λ=1 이면 이점 = 남은 보상의 합
    assert torch.allclose(buf.advantages_raw.reshape(-1), torch.tensor([3.0, 2.0, 1.0]))
    assert torch.allclose(buf.returns.reshape(-1), torch.tensor([3.0, 2.0, 1.0]))


def test_종료는_부트스트랩을_끊는다():
    buf = RolloutBuffer(3, 1, torch.device("cpu"))
    fill(buf, rewards=[1.0, 1.0, 1.0], dones=[0.0, 1.0, 0.0], values=[0.0, 0.0, 0.0])
    buf.compute_gae(last_value=torch.zeros(1), gamma=1.0, lam=1.0)
    assert torch.allclose(buf.advantages_raw.reshape(-1), torch.tensor([2.0, 1.0, 1.0]))


def test_미니배치는_모든_표본을_한_번씩():
    buf = RolloutBuffer(4, 3, torch.device("cpu"))
    fill(buf, rewards=[0.0] * 4, dones=[0.0] * 4, values=[0.0] * 4, n_envs=3)
    buf.compute_gae(last_value=torch.zeros(3))
    seen = 0
    for batch in buf.batches(5, generator=torch.Generator().manual_seed(0)):
        seen += batch[0].shape[0]
        assert batch[0].shape[1] == 73 and batch[4].dtype == torch.long
    assert seen == 12


def test_자동_리셋_한_걸음은_배치에서_빠진다():
    buf = RolloutBuffer(3, 1, torch.device("cpu"))
    # t0 정상 → t1 종료 → t2 는 자동 리셋 더미
    fill(buf, rewards=[1.0, 1.0, 0.0], dones=[0.0, 1.0, 0.0], values=[0.0, 0.0, 0.0],
         valids=[1.0, 1.0, 0.0])
    buf.compute_gae(last_value=torch.zeros(1), gamma=1.0, lam=1.0)
    rows = sum(b[0].shape[0] for b in buf.batches(100, generator=torch.Generator().manual_seed(0)))
    assert rows == 2


def test_이점은_표준화된다():
    buf = RolloutBuffer(4, 2, torch.device("cpu"))
    fill(buf, rewards=[0.0, 1.0, 2.0, 3.0], dones=[0.0] * 4, values=[0.0] * 4, n_envs=2)
    buf.compute_gae(last_value=torch.zeros(2))
    adv = torch.cat([b[6] for b in buf.batches(100, generator=torch.Generator().manual_seed(0))])
    assert abs(float(adv.mean())) < 1e-5 and abs(float(adv.std()) - 1.0) < 1e-3
