"""신호 — 경로가 지나는 신호 찾기, 신호 주기, VTD 식 알림.

VTD 는 자차 앞 신호 하나의 id·상태를 주고 교차로 안에서는 주지 않는다(주최측 답변).
신호 정지선 DB(tl_map, 214개)에는 방위가 없어 가장 가까운 도색 정지선(stoplines_all, 방위 포함)의
방위로 **같은 방향 진입로**만 고른다. 코스 H 에서 이 규칙이 신호 151(s≈96 m)을 잡는 것을 확인했다.
"""
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs

SAME_DIR_DEG = 45.0      # 도색 정지선 방위와 경로 방위 차 허용[도]
MAX_LATERAL = 8.0        # 경로에서 이만큼 안의 정지선만[m]
REPORT_RANGE = 150.0     # 이 앞까지 알린다[m] — VTD 실측 60~300 m 의 가운데쯤
PASSED_MARGIN = 5.0      # 뒷축이 정지선을 이만큼 지날 때까지는 계속 알린다[m]


@dataclass(frozen=True)
class RouteSignal:
    s: float
    tl_id: int


def route_signals(route, tl_map, stoplines_all):
    found = {}
    for tid, (x, y) in tl_map.items():
        p = route.project(x, y)
        if abs(p.lateral) > MAX_LATERAL or p.s <= 0.0 or p.s >= route.total:
            continue
        _, _, sh = min(stoplines_all, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        dh = abs((sh - route.heading_at(p.index) + math.pi) % (2.0 * math.pi) - math.pi)
        if math.degrees(dh) > SAME_DIR_DEG:
            continue
        key = round(p.s)
        if key not in found or int(tid) < found[key].tl_id:
            found[key] = RouteSignal(p.s, int(tid))
    return sorted(found.values(), key=lambda r: r.s)


@dataclass
class SignalProgram:
    mode: str = "always_green"
    green: float = 30.0
    yellow: float = 3.0
    red: float = 30.0
    offset: float = 0.0

    def state(self, t: float) -> int:
        if self.mode == "always_green":
            return rs.TL_GREEN
        tt = (t + self.offset) % (self.green + self.yellow + self.red)
        if tt < self.green:
            return rs.TL_GREEN
        if tt < self.green + self.yellow:
            return rs.TL_YELLOW
        return rs.TL_RED


def programs_for(mode, signals, rng):
    if mode == "always_green":
        return {s.tl_id: SignalProgram("always_green") for s in signals}
    if mode == "cycle":
        base = SignalProgram("cycle")
        period = base.green + base.yellow + base.red
        return {s.tl_id: SignalProgram("cycle", offset=rng.uniform(0.0, period)) for s in signals}
    raise ValueError(f"알 수 없는 신호 운용: {mode}")


class SignalReporter:
    def __init__(self, signals, lane_plan, programs):
        self.signals = signals
        self.lane_plan = lane_plan
        self.programs = programs

    def report(self, s_ego: float, index: int, t: float):
        plan = self.lane_plan[index] if 0 <= index < len(self.lane_plan) else None
        if plan and plan.get("j"):
            return -1, rs.TL_UNSET
        for sig in self.signals:
            if sig.s + PASSED_MARGIN < s_ego:
                continue
            if sig.s - s_ego <= REPORT_RANGE:
                return sig.tl_id, self.programs[sig.tl_id].state(t)
            break
        return -1, rs.TL_UNSET
