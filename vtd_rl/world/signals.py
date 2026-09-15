"""신호 — 경로가 지나는 신호 찾기, 신호 주기, VTD 식 알림.

VTD 는 자차 앞 신호 하나의 id·상태를 주고, 교차로 안에서는 신호를 주지 않는다.

경로 위 신호는 규칙 스택(`DrivingStack._signalized_lines_on_route`)과 같은 방식으로 **도색 정지선**에서 찾는다.
신호 정지선 DB(tl_map, 214개) 점은 정지선 중앙이라 차로에서 8 m 넘게 비껴 있을 수 있고 방위가 없다.
도색선(stoplines_all, 710개)은 차로마다 있고 방위가 있다. 그래서
1. 신호 DB 점이 TL_PAIR 안에 있는 도색선을 경로의 **모든 통과**에 투영해 같은 방향 통과마다 신호 하나를 둔다
   (경로가 같은 교차로를 다시 지나도 빠뜨리지 않는다).
2. 근처에 도색선이 하나도 없는 신호(예: 221)만 신호 DB 점을 경로에 투영하고, 가장 가까운 도색선 방위로
   같은 방향인지 거른다.
"""
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs

TL_PAIR = 12.0           # 도색선과 신호 DB 점이 이 안이면 그 신호의 정지선[m] — 규칙 스택 TL_ANT_PAIR
LINE_LATERAL = 4.0       # 도색선이 경로 통과에서 이 안이어야 그 통과의 선[m] — 규칙 스택 TLP_LAT
LINE_MIN_COS = 0.5       # 도색선 방위와 경로 구간 방위의 cos 하한 — 규칙 스택과 같다
SAME_LINE_S = 5.0        # 같은 신호 묶음의 선이 이 안의 s 에 여럿이면(차로별 도색선) 하나로 친다[m]
SAME_POINT = 1.0         # 신호 DB 점이 이 안에 겹치면 한 신호 묶음(예: 143·144 는 같은 자리)[m]
SAME_DIR_DEG = 45.0      # (도색선 없는 신호) 가장 가까운 도색선 방위와 경로 방위 차 허용[도]
MAX_LATERAL = 8.0        # (도색선 없는 신호) 경로에서 이만큼 안의 신호 DB 점만[m]
REPORT_RANGE = 150.0     # 이 앞까지 알린다[m] — VTD 실측 60~300 m 의 가운데쯤
PASSED_MARGIN = 5.0      # 뒷축이 정지선을 이만큼 지날 때까지는 계속 알린다[m]
NO_SIGNAL = 0            # 알릴 신호가 없을 때의 tl_id — VTD 는 -1 이 아니라 0 을 준다


@dataclass(frozen=True)
class RouteSignal:
    s: float
    tl_id: int


def _line_passes(route, x, y, heading):
    """도색선 (x, y, heading) 이 걸리는 경로 통과마다 s 하나 [s, ...].

    LINE_LATERAL 안이고 같은 방향(cos ≥ LINE_MIN_COS)인 구간이 이어진 묶음 = 한 번의 통과.
    묶음마다 선에 가장 가까운 점의 s 를 낸다.
    """
    pts, cum = route.pts, route.cum
    ch, sh = math.cos(heading), math.sin(heading)
    out, best = [], None                          # best = 지금 묶음의 (거리, s)
    for j in range(len(pts) - 1):
        seg = cum[j + 1] - cum[j]
        if seg <= 0.0:
            continue                              # 겹친 점 — 묶음을 끊지 않는다
        (x0, y0), (x1, y1) = pts[j], pts[j + 1]
        hit = None
        if (x0 - x) ** 2 + (y0 - y) ** 2 <= (LINE_LATERAL + seg) ** 2:
            dx, dy = x1 - x0, y1 - y0
            if dx * ch + dy * sh >= LINE_MIN_COS * seg:
                u = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (seg * seg)))
                d = math.hypot(x - x0 - u * dx, y - y0 - u * dy)
                if d <= LINE_LATERAL:
                    hit = (d, cum[j] + u * seg)
        if hit is None:
            if best is not None:
                out.append(best[1])
                best = None
        elif best is None or hit[0] < best[0]:
            best = hit
    if best is not None:
        out.append(best[1])
    return out


def route_signals(route, tl_map, stoplines_all):
    """경로가 지나는 신호 [RouteSignal] — s 순. 같은 신호 묶음(한 자리의 id 들)은 가장 작은 id 로 알린다."""
    cands, paired = [], set()                     # cands = (s, 신호 묶음)
    for x, y, h in stoplines_all:
        group = frozenset(int(k) for k, (px, py) in tl_map.items() if math.hypot(px - x, py - y) <= TL_PAIR)
        if not group:
            continue                              # 신호 없는 도색선
        paired |= group
        cands += [(s, group) for s in _line_passes(route, x, y, h) if 0.0 < s < route.total]

    for tid, (x, y) in tl_map.items():            # 근처에 도색선이 없는 신호 — DB 점 투영
        if int(tid) in paired:
            continue
        p = route.project(x, y)
        if abs(p.lateral) > MAX_LATERAL or p.s <= 0.0 or p.s >= route.total:
            continue
        _, _, lh = min(stoplines_all, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        dh = abs((lh - route.heading_at(p.index) + math.pi) % (2.0 * math.pi) - math.pi)
        if math.degrees(dh) > SAME_DIR_DEG:
            continue
        group = frozenset(int(k) for k, (px, py) in tl_map.items() if math.hypot(px - x, py - y) <= SAME_POINT)
        cands.append((p.s, group))

    kept = []                                     # (s, 묶음) — 묶음마다 SAME_LINE_S 안의 첫 선만
    for s, group in sorted(cands, key=lambda c: (c[0], min(c[1]))):
        if not any(g == group and s - s0 <= SAME_LINE_S for s0, g in kept):
            kept.append((s, group))
    return [RouteSignal(s, min(group)) for s, group in kept]


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
        self.signals = sorted(signals, key=lambda r: r.s)     # report 는 앞에서부터 훑는다
        self.lane_plan = lane_plan
        self.programs = programs

    def report(self, s_ego: float, index: int, t: float):
        plan = self.lane_plan[index] if 0 <= index < len(self.lane_plan) else None
        if plan and plan.get("j"):
            return NO_SIGNAL, rs.TL_UNSET
        for sig in self.signals:
            if sig.s + PASSED_MARGIN < s_ego:
                continue
            if sig.s - s_ego <= REPORT_RANGE:
                return sig.tl_id, self.programs[sig.tl_id].state(t)
            break
        return NO_SIGNAL, rs.TL_UNSET
