"""학생 정책망 — DeepSets 물체 인코더 + 몸통 + 두 머리(연속 조향·가속, 범주형 지시등).

물체는 공유 MLP 를 거쳐 **마스크를 씌운 뒤 최댓값으로** 합친다. 개수·순서가 달라져도 결과가 같아야
한다(스펙 §4.1). 머리 모양은 M4 의 PPO 가 그대로 쓴다 — 가우시안 평균·로그표준편차와 범주형 로짓.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from vtd_rl.env.action import TURNS
from vtd_rl.policy.encode import OBJ_DIM, VEC_DIM, flatten_obs, to_tensors

NEG_BIG = -1.0e9          # 마스크된 자리를 최댓값 합치기에서 제외하는 값


@dataclass(frozen=True)
class PolicyConfig:
    obj_hidden: int = 64
    obj_out: int = 64
    trunk: tuple = (256, 256)
    log_std_init: float = -1.0
    log_std_min: float = -2.0       # 조향 표준편차가 무너져 가속 머리를 굶기지 않도록(σ≥0.135, ~8배 차이로 제한)


class DrivePolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        self.obj = nn.Sequential(nn.Linear(OBJ_DIM, cfg.obj_hidden), nn.Tanh(),
                                 nn.Linear(cfg.obj_hidden, cfg.obj_out), nn.Tanh())
        layers, last = [], VEC_DIM + cfg.obj_out
        for width in cfg.trunk:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last, 2)
        self.turn = nn.Linear(last, len(TURNS))
        self.log_std = nn.Parameter(torch.full((2,), float(cfg.log_std_init)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, vec, objs, mask):
        h = self.obj(objs)                                   # [B, N, obj_out]
        h = h.masked_fill(mask[..., None] <= 0.0, NEG_BIG)
        pooled = h.max(dim=1).values
        pooled = torch.where(mask.sum(dim=1, keepdim=True) > 0.0, pooled,
                             torch.zeros_like(pooled))        # 물체가 없으면 0
        z = self.trunk(torch.cat([vec, pooled], dim=-1))
        log_std = self.log_std.clamp(min=self.cfg.log_std_min)
        return self.mean(z), log_std, self.turn(z)

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = True, generator=None) -> dict:
        vec, objs, mask = to_tensors(*flatten_obs(obs), self.device)
        mean, log_std, logits = self(vec, objs, mask)
        if deterministic:
            control = mean[0]
            turn = int(torch.argmax(logits[0]).item())
        else:
            # 표본 추출은 CPU 에서 한다 — CPU 생성기를 CUDA 텐서에 쓰면 오류가 난다
            noise = torch.randn(mean.shape, generator=generator).to(mean.device)
            control = mean[0] + noise[0] * log_std.exp()
            probs = torch.softmax(logits[0], dim=-1).cpu()
            turn = int(torch.multinomial(probs, 1, generator=generator).item())
        control = control.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return {"control": control, "turn": turn}

    def save(self, path: str):
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "DrivePolicy":
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = PolicyConfig(**{**blob["cfg"], "trunk": tuple(blob["cfg"]["trunk"])})
        net = cls(cfg)
        net.load_state_dict(blob["state"])
        if device is not None:
            net.to(device)
        net.eval()
        return net
