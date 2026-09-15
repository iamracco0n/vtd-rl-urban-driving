"""액터 — 규칙 스택 오프라인 시뮬(`eval/mock_vtd.py`)의 등장·운동 모델을 그대로 쓴다.

호출 순서도 오프라인 시뮬과 같다: 등장 판정 -> (처음이면) 상태 초기화 -> 한 스텝 적분 -> Obj.
"""
import math

from vtd_rl import rule_stack as rs

_mv = rs.mock_vtd


class ActorSet:
    def __init__(self, actors):
        self.actors = list(actors)
        self.reset()

    def reset(self):
        self._state = {}
        self._spawn_t = {}

    def step(self, t, ex, ey, dt):
        objs = []
        for a in self.actors:
            if not _mv.spawned(a, t, ex, ey, self._spawn_t):
                continue
            st = self._state.get(a.id)
            if st is None:
                st = self._state[a.id] = _mv.init_actor_state(a)
            _mv.step_actor(a, st, ex, ey, dt)
            objs.append(rs.Obj(a.id, st["x"], st["y"], 0.0, st["hd"], st["sp"],
                               a.size[0], a.size[1], a.size[2]))
        return objs


def nearest_objects(objs, ex, ey, max_range, max_n):
    """VTD 9910 처럼 수평 max_range 안을 가까운 순으로 max_n 개."""
    ranked = sorted(((math.hypot(o.x - ex, o.y - ey), o) for o in objs), key=lambda p: p[0])
    return [o for d, o in ranked if d <= max_range][:max_n]
