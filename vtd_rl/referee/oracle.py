"""사후 채점기 — score_fma.main() 과 같은 조립으로 한 판을 채점하고 감점 호출을 기록한다.

심판(온라인)이 맞는지 가르는 정답이다. main() 과 같은 데이터를 쓴다:
경로 = 판의 ego_route, 제한속도 = 차로계획 lim, 차로변경 표식 = ego_lanes[i][4],
정지선 = routes/tl_map_livinglab.json(문자열 키 그대로), 횡단보도 = routes/crosswalks.json 중 경로 6 m 안.
"""
import copy
import functools
import json
import math
from dataclasses import dataclass

from vtd_rl import rule_stack as rs

sf = rs.score_fma


class _RecordingSheet(sf.Sheet):
    def __init__(self, n):
        super().__init__(n)
        self.calls = []

    def hit(self, sec, item, level, why):
        self.calls.append((sec, item, level))
        super().hit(sec, item, level, why)


@dataclass
class MatchInputs:
    route: list
    lims: list
    lc_flag: list
    tl_stops: dict
    cws: list


@dataclass
class OracleResult:
    hits: list
    state: list
    respawns: dict
    notes: list


@functools.lru_cache(maxsize=None)
def _json(rel):
    with open(rs.path(rel), encoding="utf-8") as f:
        return json.load(f)


def match_inputs(board) -> MatchInputs:
    route = [list(p) for p in board.route.pts]
    lims = [p.get("lim") for p in board.lane_plan]
    lc_flag = [bool(p[4]) if len(p) > 4 else False for p in board.ego_lanes]
    cws = [c for c in _json("routes/crosswalks.json")["crosswalks"]      # default=99·6.0: score_fma.main 안의 숫자[m]
           if min((math.hypot(p[0] - c["x"], p[1] - c["y"]) for p in route), default=99) < 6.0]
    return MatchInputs(route, lims, lc_flag, _json("routes/tl_map_livinglab.json"), cws)


def make_lim_at(secs, lims):
    def lim_at(x, y):
        if not lims:
            return None
        i = secs.index_of(x, y)
        return lims[i] if i < len(lims) else None
    return lim_at


def make_lc_at(secs, lc_flag):
    def lc_at(x, y):
        if not lc_flag:
            return False
        i = secs.index_of(x, y)
        lo = max(0, i - int(sf.LC_MARGIN_M))
        hi = min(len(lc_flag), i + int(sf.LC_MARGIN_M) + 1)
        return any(lc_flag[lo:hi])
    return lc_at


def score_episode(rows, board, sections=5, use_map=True) -> OracleResult:
    rows = copy.deepcopy(list(rows))
    m = match_inputs(board)
    secs = sf.Sections(m.route, sections)
    sheet = _RecordingSheet(secs.n)
    if use_map:
        rs.check_lanes.classify(rows, rs.load_map())
    notes = []
    sf.item_speed(rows, sheet, secs, make_lim_at(secs, m.lims))
    if use_map:
        sf.item_lane_geometry(rows, sheet, secs, make_lc_at(secs, m.lc_flag))
    sf.item_traffic_light(rows, sheet, secs, m.tl_stops, notes)
    sf.item_contact(rows, sheet, secs)
    sf.item_pedestrian(rows, sheet, secs)
    sf.item_crosswalk_stop(rows, sheet, secs, m.cws)
    sf.item_turn_signal(rows, sheet, secs)
    respawns = sf.item_respawn(rows, sheet, secs)
    return OracleResult(sheet.calls, [dict(s) for s in sheet.state], respawns, notes)
