"""PPO 손실 + 모방 손실(감쇠) — 스펙 §6.3.

모방 손실은 M3 라벨(선생님)에 대한 로그가능도다. 처음에는 강하게 잡아 두고 반감기로 줄인다 —
PPO 단계에서는 선생님을 다시 돌리지 않고 모아 둔 데이터만 쓴다(스펙 §6.3).
"""
from dataclasses import dataclass

import torch
from torch import nn

from vtd_rl.policy.train import TrainConfig, policy_loss


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    imitation_coef0: float = 1.0
    imitation_half_life: int = 2_000_000
    target_kl: float = 0.03


def imitation_coef(step: int, cfg: PPOConfig) -> float:
    return cfg.imitation_coef0 * 0.5 ** (step / max(cfg.imitation_half_life, 1))


def ppo_losses(net, batch, cfg: PPOConfig):
    """잘라낸 정책 손실 + 잘라낸 가치 손실 + 엔트로피 보너스.

    비율은 저장해 둔 **원표본**(`raw`)으로 계산한다 — 자른 값(`control`)으로 계산하면
    분포가 안 맞는다(`DrivePolicy.sample()`의 계약, 스펙 §6.3).
    """
    vec, objs, mask, raw, turn, old_log_prob, adv, ret, old_value = batch
    log_prob, entropy, value = net.evaluate_actions(vec, objs, mask, raw, turn)
    ratio = (log_prob - old_log_prob).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
    policy = -torch.min(unclipped, clipped).mean()
    v_clipped = old_value + (value - old_value).clamp(-cfg.clip, cfg.clip)
    value_loss = 0.5 * torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
    ent = entropy.mean()
    loss = policy + cfg.value_coef * value_loss - cfg.entropy_coef * ent
    with torch.no_grad():
        approx_kl = float(((ratio - 1.0) - (log_prob - old_log_prob)).mean())
        clip_frac = float(((ratio - 1.0).abs() > cfg.clip).float().mean())
    return loss, {"policy": policy.item(), "value": value_loss.item(), "entropy": ent.item(),
                  "approx_kl": approx_kl, "clip_frac": clip_frac}


def imitation_loss(net, dagger_batch, cfg: PPOConfig):
    """M3 라벨에 대한 로그가능도 — `vtd_rl.policy.train.policy_loss` 를 그대로 쓴다.

    `net` 은 `ActorCritic`(가치 머리 포함)이지만, `policy_loss` 는 `DrivePolicy` 를
    직접 호출하므로(`net(vec, objs, mask) -> mean, log_std, logits`) 반드시 `net.policy` 를
    넘긴다.
    """
    return policy_loss(net.policy, dagger_batch, TrainConfig())


def dagger_batches(dataset, batch_size: int, generator=None, device=None):
    """M3 라벨을 끝없이 돌린다 — 롤아웃이 수백 번이라 한 바퀴로는 모자란다."""
    while True:
        for batch in dataset.batches(batch_size, generator=generator, device=device):
            yield batch


def update(net, opt, buffer, dagger_iter, cfg: PPOConfig, step: int, generator=None) -> dict:
    """에폭을 반복하며 PPO 손실 + 모방 손실(감쇠)을 합쳐 한 걸음씩 최적화한다.

    최적화 한 걸음마다 `net.clamp_log_std()` 를 불러 표준편차 범위를 지킨다(M3 가 σ 붕괴로
    학생이 브레이크만 밟았던 실패를 되풀이하지 않는다). `approx_kl` 이 `target_kl` 을 넘으면
    그 에폭에서 바로 멈춘다.
    """
    coef = imitation_coef(step, cfg)
    sums, count = {}, 0
    for _epoch in range(cfg.epochs):
        stop = False
        for batch in buffer.batches(cfg.minibatch, generator=generator):
            loss, parts = ppo_losses(net, batch, cfg)
            parts["imitation"] = 0.0
            if dagger_iter is not None and coef > 1e-4:
                dbatch = next(dagger_iter)
                imi, iparts = imitation_loss(net, dbatch, cfg)
                loss = loss + coef * imi
                parts["imitation"] = iparts["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            net.clamp_log_std()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            if parts["approx_kl"] > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: v / max(count, 1) for k, v in sums.items()}
    out["imitation_coef"] = coef
    out["updates"] = count
    return out
