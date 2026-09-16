"""행동 — 스펙 §4.2. 조향·목표가속 연속값 [-1, 1] + 지시등 범주형, 판단 10 Hz.

가속은 0 을 가운데 두고 양쪽 한계가 다르다(+2 / −5). 그래서 부호별로 따로 늘린다 —
한 축으로 선형 변환하면 '가속 0' 이 행동 0 이 아니게 되어 정책이 매 프레임 미세 가속을 배운다.
"""
import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from vtd_rl import rule_stack as rs

TURNS = (rs.TS_OFF, rs.TS_LEFT, rs.TS_RIGHT)


@dataclass(frozen=True)
class ActionConfig:
    decision_hz: float = 10.0
    max_steer: float = math.radians(35.0)     # 규칙 스택 PurePursuit 와 같은 한계
    accel_min: float = -5.0                   # 규칙 스택 LongPI 와 같은 한계
    accel_max: float = 2.0


def action_space(cfg: ActionConfig = ActionConfig()) -> spaces.Dict:
    return spaces.Dict({
        "control": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "turn": spaces.Discrete(len(TURNS)),
    })


def to_command(action, cfg: ActionConfig = ActionConfig()):
    control = np.clip(np.asarray(action["control"], dtype=np.float32), -1.0, 1.0)
    steer = float(control[0]) * cfg.max_steer
    a = float(control[1])
    accel = a * cfg.accel_max if a >= 0.0 else a * abs(cfg.accel_min)
    k = int(action["turn"])
    # 음수 색인은 파이썬이 조용히 뒤에서부터 센다(-1 -> TS_RIGHT). 지시등이 반대로 켜지는 걸
    # 나중에 주행 로그에서 찾는 것보다 여기서 소리 내어 터지는 게 낫다.
    if not 0 <= k < len(TURNS):
        raise ValueError(f"지시등 색인은 0~{len(TURNS) - 1}: {k}")
    return steer, accel, TURNS[k]


def from_command(steer: float, accel: float, turn: int, cfg: ActionConfig = ActionConfig()):
    a = accel / cfg.accel_max if accel >= 0.0 else accel / abs(cfg.accel_min)
    control = np.clip(np.asarray([steer / cfg.max_steer, a], dtype=np.float32), -1.0, 1.0)
    return {"control": control, "turn": TURNS.index(int(turn)) if int(turn) in TURNS else 0}


def frames_per_step(cfg: ActionConfig, dt: float) -> int:
    return max(1, round(1.0 / (cfg.decision_hz * dt)))
