"""행동 복제 학습 — 조향·가속은 가우시안 음의 로그가능도, 지시등은 교차엔트로피.

M4 의 PPO 가 같은 분포를 쓰므로 여기서도 평균제곱오차가 아니라 로그가능도로 배운다
(표준편차까지 배워 두면 PPO 시작점이 자연스럽다).
"""
import math
import time
from dataclasses import dataclass

import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 8
    turn_weight: float = 0.5        # 지시등은 대부분 '끔' 이라 가중치를 낮춘다
    grad_clip: float = 1.0
    seed: int = 0


def policy_loss(net, batch, cfg: TrainConfig):
    vec, objs, mask, control, turn = batch
    mean, log_std, logits = net(vec, objs, mask)
    var = (2.0 * log_std).exp()
    nll = 0.5 * (((control - mean) ** 2) / var + 2.0 * log_std + LOG_2PI)
    control_loss = nll.sum(dim=-1).mean()
    turn_loss = nn.functional.cross_entropy(logits, turn)
    total = control_loss + cfg.turn_weight * turn_loss
    return total, {"control": float(control_loss.item()), "turn": float(turn_loss.item()),
                   "total": float(total.item())}


def train_epochs(net, dataset, cfg: TrainConfig = TrainConfig(), device=None, log=None) -> dict:
    device = device or net.device
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    t0, last = time.perf_counter(), {}
    for epoch in range(cfg.epochs):
        sums, batches = {"total": 0.0, "control": 0.0, "turn": 0.0}, 0
        for batch in dataset.batches(cfg.batch_size, generator=gen, device=device):
            loss, parts = policy_loss(net, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            for k in sums:
                sums[k] += parts[k]
            batches += 1
        last = {"epoch": epoch, "loss": sums["total"] / max(batches, 1),
                "control": sums["control"] / max(batches, 1),
                "turn": sums["turn"] / max(batches, 1)}
        if log is not None:
            log(last)
    net.eval()
    return {"epochs": cfg.epochs, "samples": len(dataset), "loss": last.get("loss", float("nan")),
            "control": last.get("control", float("nan")), "turn": last.get("turn", float("nan")),
            "seconds": time.perf_counter() - t0}


@torch.no_grad()
def evaluate_labels(net, dataset, device=None, cfg: TrainConfig = TrainConfig()) -> dict:
    device = device or net.device
    net.to(device).eval()
    vec, objs, mask, control, turn = next(iter(dataset.batches(len(dataset), device=device)))
    mean, log_std, logits = net(vec, objs, mask)
    loss, parts = policy_loss(net, (vec, objs, mask, control, turn), cfg)
    return {"loss": parts["total"], "control": parts["control"], "turn": parts["turn"],
            "control_mae": float((mean - control).abs().mean().item()),
            "turn_acc": float((logits.argmax(dim=-1) == turn).float().mean().item())}
