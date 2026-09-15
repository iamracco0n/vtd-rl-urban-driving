"""판 = 규칙 스택 Scenario + 경로점별 차로계획 + 신호 운용 방식."""
import bisect
import json
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.world.route import RouteIndex

SIGNAL_MODES = ("always_green", "cycle")


@dataclass
class Board:
    name: str
    scenario: object
    lane_plan: list
    signals: str = "always_green"
    route: RouteIndex = field(init=False)

    def __post_init__(self):
        if self.signals not in SIGNAL_MODES:
            raise ValueError(f"신호 운용은 {SIGNAL_MODES} 중 하나: {self.signals}")
        pts = self.scenario.ego_route or self.scenario.route_points()
        if len(pts) != len(self.lane_plan):
            # 차로계획은 원본 ego_route 와 짝이다. route_points() 로 촘촘하게 나누면
            # 코스 H 는 2021 점이 되어 6점 어긋난다(2026-09-15 확인).
            raise ValueError(f"{self.name}: 경로점 {len(pts)} != 차로계획 {len(self.lane_plan)}")
        self.route = RouteIndex(pts)

    @property
    def start_pose(self):
        x, y = self.route.pts[0]
        return x, y, self.route.heading_at(0)

    @property
    def goal(self):
        return self.route.pts[-1]


def _load_json(rel):
    with open(rs.path(rel), encoding="utf-8") as f:
        return json.load(f)


def load_board(entry: dict, signals: str = "always_green") -> Board:
    sc = rs.Scenario.load(rs.path(entry["route"]))
    lane = _load_json(entry["lane"])["pts"]
    return Board(entry["name"], sc, lane, signals)


def slice_board(board: Board, s_from: float, s_to: float, name: str) -> Board:
    cum = board.route.cum
    i0 = bisect.bisect_left(cum, s_from)
    i1 = bisect.bisect_right(cum, s_to) - 1
    if i1 - i0 < 2:
        raise ValueError(f"자른 구간이 너무 짧다: {s_from}~{s_to}")
    pts = [list(p) for p in board.route.pts[i0:i1 + 1]]
    length = cum[i1] - cum[i0]
    sc0 = board.scenario
    sc = rs.Scenario(
        name=name,
        ego_start=[pts[0][0], pts[0][1], board.route.heading_at(i0)],
        ego_goal=pts[-1],
        speed_limit=sc0.speed_limit,
        duration=max(60.0, length / 5.0 + 30.0),
        actors=[], lights=[], zones=[],
        ego_route=pts, respawns=[], tl_stops={},
    )
    return Board(name, sc, board.lane_plan[i0:i1 + 1], board.signals)


def load_curriculum(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["name"], [load_board(e, d["signals"]) for e in d["boards"]]
