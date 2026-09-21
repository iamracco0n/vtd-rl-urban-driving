import os

import pytest
import torch

from vtd_rl.policy.net import DrivePolicy
from vtd_rl.rl.actor_critic import ActorCritic

M3_CKPT = "runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt"


def test_행동과_가치를_함께_낸다(small_ac, obs_batch):
    ac = small_ac()
    vec, objs, mask = obs_batch(5)
    out = ac.act(vec, objs, mask, generator=torch.Generator().manual_seed(0))
    assert out["value"].shape == (5,) and out["log_prob"].shape == (5,)
    lp, ent, val = ac.evaluate_actions(vec, objs, mask, out["raw"], out["turn"])
    assert torch.allclose(lp, out["log_prob"], atol=1e-5) and val.shape == (5,)
    assert torch.all(ent > 0.0)


def test_가치_학습이_정책_출력을_바꾸지_않는다(small_ac, obs_batch):
    ac = small_ac()
    vec, objs, mask = obs_batch(4)
    before = ac.policy(vec, objs, mask)[0].detach().clone()
    loss = ac.value(vec, objs, mask).pow(2).mean()
    loss.backward()
    torch.optim.SGD(ac.parameters(), lr=0.1).step()
    after = ac.policy(vec, objs, mask)[0].detach()
    assert torch.allclose(before, after, atol=1e-6)     # 가치 경로가 정책 몸통과 분리돼 있다


def test_저장하고_불러오면_같은_행동(small_ac, obs_batch, tmp_path):
    ac = small_ac()
    path = str(tmp_path / "ac.pt")
    ac.save(path)
    other = ActorCritic.load(path)
    vec, objs, mask = obs_batch(3)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    a, b = ac.act(vec, objs, mask, generator=g1), other.act(vec, objs, mask, generator=g2)
    assert torch.allclose(a["raw"], b["raw"], atol=1e-6) and torch.equal(a["turn"], b["turn"])


@pytest.mark.skipif(not os.path.exists(M3_CKPT), reason="M3 체크포인트가 있어야 한다")
def test_M3_체크포인트를_이식한다(obs_batch):
    student = DrivePolicy.load(M3_CKPT)
    ac = ActorCritic.from_policy(M3_CKPT)
    vec, objs, mask = obs_batch(4)
    # 이식 뒤 log_std 를 범위 안으로 넣으므로(-2.0024 → -2.0) 평균만 같은지 본다
    assert torch.allclose(ac.policy(vec, objs, mask)[0], student(vec, objs, mask)[0], atol=1e-6)
    assert ac.policy.cfg.trunk == student.cfg.trunk
    assert ac.value(vec, objs, mask).shape == (4,)
