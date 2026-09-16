"""관측 — 스펙 §4.1. 자차 기준 좌표, 정규화, float32.

2부에서 영상 인코더로 갈아 끼울 수 있게 관측 생성은 이 모듈 하나에 모은다.
물체는 (개수 고정 + 마스크)로 내보내고, DeepSets 합치기는 정책이 한다.
"""
import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from vtd_rl import rule_stack as rs
from vtd_rl.env.board_index import board_index

TL_STATES = 7             # vtd_io: UNSET·RED·YELLOW·GREEN·LEFT·GREEN_LEFT·FLASH
PED_L, PED_W = rs.score_fma.PED_L, rs.score_fma.PED_W     # 사람·이륜차 판별(채점기와 같은 치수)
PED_MIN_H = rs.score_fma.PED_MIN_H


@dataclass(frozen=True)
class ObsConfig:
    objects: int = 16
    route_points: int = 20
    route_ahead: float = 50.0        # 경로점을 이만큼 앞까지 고르게 뽑는다[m]
    v_max: float = 25.0              # 속도 정규화[m/s] (90 km/h)
    yaw_max: float = 1.0             # 요레이트 정규화[rad/s]
    lat_max: float = 5.0             # 횡오프셋 정규화[m]
    room_max: float = 8.0            # 차로 여유 정규화[m]
    dist_max: float = 100.0          # 앞쪽 거리 정규화[m]
    remain_max: float = 1000.0       # 남은 거리 정규화[m]
    obj_x: float = 80.0              # 물체 전후 정규화[m] — World.object_range(9910 수평 범위)와 맞춘다
    obj_y: float = 20.0              # 물체 좌우 정규화[m]
    obj_size: float = 12.0           # 물체 치수 정규화[m]


def observation_space(cfg: ObsConfig = ObsConfig()) -> spaces.Dict:
    def box(shape, lo=-1.0, hi=1.0):
        return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)
    return spaces.Dict({
        "ego": box((9,)),
        "route": box((cfg.route_points, 2)),
        "nav": box((3,)),
        "plan": box((10,)),
        "signal": box((TL_STATES + 4,)),
        "objects": box((cfg.objects, 12)),
        "object_mask": box((cfg.objects,), 0.0, 1.0),
    })


def _clip(value, scale):
    return float(np.clip(value / scale, -1.0, 1.0))


def _turn_onehot(turn):
    out = [0.0, 0.0, 0.0]
    out[int(turn) if int(turn) in (0, 1, 2) else 0] = 1.0
    return out


def _object_class(o):
    """치수로 종류를 나눈다 — 채점기 ⑪⑭ 와 같은 기준(차량·사람류·사물)."""
    if o.length > PED_L:
        return [1.0, 0.0, 0.0]
    if o.length <= PED_L and o.width <= PED_W and o.height >= PED_MIN_H:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def build_observation(world, state, info, prev_action, cfg: ObsConfig = ObsConfig()) -> dict:
    board = world.board
    idx = board_index(board)
    route = board.route
    ego = world.ego
    s = info.s if info is not None else 0.0
    lateral = info.lateral if info is not None else 0.0
    index = info.index if info is not None else 0
    plan = idx.plan_at(index)
    lim = plan.get("lim") or 0.0
    yaw_rate = ego.v * math.tan(ego.steer) / world.cfg.dynamics.wheelbase

    ego_vec = [_clip(ego.v, cfg.v_max), float(np.clip(prev_action[0], -1.0, 1.0)),
               float(np.clip(prev_action[1], -1.0, 1.0)), _clip(yaw_rate, cfg.yaw_max)]
    ego_vec += _turn_onehot(prev_action[2])
    ego_vec += [_clip(lim, cfg.v_max), _clip(ego.v - lim, cfg.v_max)]

    cos_h, sin_h = math.cos(-ego.heading), math.sin(-ego.heading)
    pts = np.zeros((cfg.route_points, 2), dtype=np.float32)
    step = cfg.route_ahead / cfg.route_points
    for k in range(cfg.route_points):
        px, py, _ph = route.point_at(min(s + step * (k + 1), route.total))
        dx, dy = px - ego.x, py - ego.y
        pts[k, 0] = _clip(dx * cos_h - dy * sin_h, cfg.route_ahead)
        pts[k, 1] = _clip(dx * sin_h + dy * cos_h, cfg.route_ahead)

    heading_err = (route.heading_at(index) - ego.heading + math.pi) % (2.0 * math.pi) - math.pi
    nav = [_clip(lateral, cfg.lat_max), _clip(heading_err, math.pi),
           _clip(route.total - s, cfg.remain_max)]

    sig_dir = plan.get("sig") or 0
    plan_vec = [_clip(plan.get("w") or 0.0, cfg.room_max),
                _clip(plan.get("l") or 0.0, cfg.room_max),
                _clip(plan.get("r") or 0.0, cfg.room_max),
                _clip(plan.get("xl") or 0.0, cfg.room_max),
                _clip(plan.get("xr") or 0.0, cfg.room_max),
                _clip(plan.get("need") or 0.0, cfg.lat_max),
                1.0 if plan.get("j") else 0.0,
                1.0 if sig_dir > 0 else 0.0,
                1.0 if sig_dir < 0 else 0.0,
                _clip(idx.ahead(s, "signal"), cfg.dist_max)]

    signal = [0.0] * TL_STATES
    tl_state = int(state.tl_state) if 0 <= int(state.tl_state) < TL_STATES else 0
    signal[tl_state if state.tl_id > 0 else 0] = 1.0
    signal += [_clip(idx.ahead(s, "signal"), cfg.dist_max),
               _clip(idx.ahead(s, "stopline"), cfg.dist_max),
               _clip(idx.ahead(s, "crosswalk"), cfg.dist_max),
               1.0 if (index < len(idx.zone) and idx.zone[index]) else 0.0]

    objs = np.zeros((cfg.objects, 12), dtype=np.float32)
    mask = np.zeros((cfg.objects,), dtype=np.float32)
    near = sorted(state.objects, key=lambda o: math.hypot(o.x - ego.x, o.y - ego.y))
    for k, o in enumerate(near[:cfg.objects]):
        dx, dy = o.x - ego.x, o.y - ego.y
        fx = dx * cos_h - dy * sin_h
        fy = dx * sin_h + dy * cos_h
        dh = (o.heading - ego.heading + math.pi) % (2.0 * math.pi) - math.pi
        p = route.project(o.x, o.y, hint=index)
        objs[k] = [_clip(fx, cfg.obj_x), _clip(fy, cfg.obj_y), math.cos(dh), math.sin(dh),
                   _clip(o.speed, cfg.v_max), _clip(o.length, cfg.obj_size),
                   _clip(o.width, cfg.obj_size), _clip(o.height, cfg.obj_size),
                   _clip(p.lateral, cfg.lat_max)] + _object_class(o)
        mask[k] = 1.0

    return {
        "ego": np.asarray(ego_vec, dtype=np.float32),
        "route": pts,
        "nav": np.asarray(nav, dtype=np.float32),
        "plan": np.asarray(plan_vec, dtype=np.float32),
        "signal": np.asarray(signal, dtype=np.float32),
        "objects": objs,
        "object_mask": mask,
    }
