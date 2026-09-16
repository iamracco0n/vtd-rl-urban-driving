"""판마다 한 번 만드는 색인 — 관측이 쓰는 '앞에 무엇이 얼마나 남았나'.

리셋마다 다시 계산하면 판당 수십 ms 가 그냥 나간다(횡단보도를 경로 전체와 비교한다).
"""
import bisect
import math

from vtd_rl import rule_stack as rs
from vtd_rl.world.signals import route_signals, LINE_MIN_COS

NEAR_ROUTE = 8.0          # 경로에서 이 안에 있는 것만 경로 위의 것으로 본다[m]
ZONE_LIM = rs.score_fma.ZONE_LIM


class BoardIndex:
    def __init__(self, board):
        self.board = board
        db = rs.map_db()
        route = board.route
        self.signals = route_signals(route, db["tl_map"], db["stoplines_all"])
        self.signal_s = sorted(sig.s for sig in self.signals)
        self.crosswalk_s = self._project(route, [(c["x"], c["y"]) for c in db["crosswalks"]])
        self.stopline_s = self._project_stoplines(route, db["stoplines_all"])
        self.zone = [bool(p and (p.get("lim") or 99.0) <= ZONE_LIM) for p in board.lane_plan]
        self._kinds = {"signal": self.signal_s, "crosswalk": self.crosswalk_s,
                       "stopline": self.stopline_s}

    @staticmethod
    def _project(route, points):
        out = []
        for x, y in points:
            p = route.project(x, y)
            if abs(p.lateral) <= NEAR_ROUTE and 0.0 < p.s < route.total:
                out.append(p.s)
        return sorted(out)

    @staticmethod
    def _project_stoplines(route, stoplines_with_heading):
        """도색 정지선을 경로에 투영하되, 진행 방향이 같은 것만 (신호.py 와 같은 논리).

        신호 정지선은 경로의 다른 차선에도 있고, 반대 차선 정지선을 '앞에 온다'로 보면 잘못된 판단을 한다.
        """
        out = []
        for x, y, heading in stoplines_with_heading:
            p = route.project(x, y)
            if abs(p.lateral) <= NEAR_ROUTE and 0.0 < p.s < route.total:
                # 경로의 이 지점 방향과 정지선 방향이 같은지 확인
                route_heading = route.heading_at(p.index)
                heading_diff = heading - route_heading
                # 각도 차이를 [-π, π] 범위로 정규화
                heading_diff = (heading_diff + math.pi) % (2.0 * math.pi) - math.pi
                if math.cos(heading_diff) >= LINE_MIN_COS:
                    out.append(p.s)
        return sorted(out)

    def ahead(self, s: float, kind: str) -> float:
        xs = self._kinds[kind]
        i = bisect.bisect_left(xs, s)
        return xs[i] - s if i < len(xs) else math.inf

    def plan_at(self, index: int) -> dict:
        plan = self.board.lane_plan[index] if 0 <= index < len(self.board.lane_plan) else None
        return plan or {}


_CACHE: dict = {}


def clear_board_index_cache():
    _CACHE.clear()


def board_index(board) -> BoardIndex:
    key = (board.name, len(board.route.pts))
    idx = _CACHE.get(key)
    if idx is None:
        idx = _CACHE[key] = BoardIndex(board)
    return idx
