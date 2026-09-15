import math

import pytest

from vtd_rl.world.dynamics import DynamicsParams, EgoState, step

DT = 0.05


def run(st, steer, accel, seconds, p):
    for _ in range(int(round(seconds / DT))):
        st = step(st, steer, accel, DT, p)
    return st


def test_지연_없는_직선_가속은_규칙_스택_오프라인_시뮬과_같다():
    p = DynamicsParams(accel_tau=0.0, max_steer_rate=100.0)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 1.0, p)
    assert st.v == pytest.approx(2.0)
    assert st.x == pytest.approx(1.05)       # sum_{k=1..20} (0.1k) * 0.05
    assert st.y == pytest.approx(0.0)


def test_가속_1차_지연():
    p = DynamicsParams(accel_tau=0.25)
    st = run(EgoState(0.0, 0.0, 0.0), 0.0, 2.0, 0.25, p)
    assert st.accel == pytest.approx(2.0 * (1.0 - math.exp(-1.0)), rel=1e-6)


def test_조향_속도_제한():
    p = DynamicsParams(max_steer_rate=1.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.5, 0.0, 0.1, p)
    assert st.steer == pytest.approx(0.1)


def test_조향_한계():
    p = DynamicsParams(max_steer_rate=100.0)
    st = step(EgoState(0.0, 0.0, 0.0), 2.0, 0.0, DT, p)
    assert st.steer == pytest.approx(math.radians(35.0))


def test_가속_범위와_후진_없음():
    p = DynamicsParams(accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=1.0), 0.0, -9.0, 1.0, p)
    assert st.accel == pytest.approx(-5.0)
    assert st.v == 0.0 and st.x < 0.2


def test_좌회전은_왼쪽으로_간다():
    p = DynamicsParams(max_steer_rate=100.0, accel_tau=0.0)
    st = run(EgoState(0.0, 0.0, 0.0, v=5.0), 0.3, 0.0, 1.0, p)
    assert st.heading > 0.3 and st.y > 0.5
