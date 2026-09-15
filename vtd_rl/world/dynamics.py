"""자차 동역학 — 자전거 모델 + 가속 1차 지연 + 조향 속도 제한.

적분 순서는 규칙 스택 오프라인 시뮬(`eval/mock_vtd.py`)과 같다:
속도 -> 방위(새 속도로) -> 위치(새 방위로). 지연·속도제한을 끄면 결과가 같다.
"""
import math
from dataclasses import dataclass


@dataclass
class DynamicsParams:
    wheelbase: float = 2.95
    max_steer: float = math.radians(35.0)
    max_steer_rate: float = 1.0      # rad/s — M5 보정 전 초기값
    accel_tau: float = 0.25          # s — 0 이면 즉시 반영. M5 보정 전 초기값
    accel_min: float = -5.0
    accel_max: float = 2.0


@dataclass
class EgoState:
    x: float
    y: float
    heading: float
    v: float = 0.0
    accel: float = 0.0
    steer: float = 0.0


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def step(st: EgoState, steer_cmd: float, accel_cmd: float, dt: float, p: DynamicsParams) -> EgoState:
    steer_cmd = _clamp(steer_cmd, -p.max_steer, p.max_steer)
    dmax = p.max_steer_rate * dt
    steer = st.steer + _clamp(steer_cmd - st.steer, -dmax, dmax)

    accel_cmd = _clamp(accel_cmd, p.accel_min, p.accel_max)
    if p.accel_tau <= 0.0:
        accel = accel_cmd
    else:
        accel = st.accel + (accel_cmd - st.accel) * (1.0 - math.exp(-dt / p.accel_tau))

    v = max(0.0, st.v + accel * dt)
    heading = st.heading + v / p.wheelbase * math.tan(steer) * dt
    heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
    x = st.x + v * math.cos(heading) * dt
    y = st.y + v * math.sin(heading) * dt
    return EgoState(x, y, heading, v, accel, steer)
