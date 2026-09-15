"""온라인 심판 뼈대.

- SpanTracker: score_fma.spans() 를 한 행씩. 참이 이어지다 거짓 행이 오면 그 구간을 닫는다(길이 조건 같음).
- Context: score_fma.main() 과 같은 보조 데이터(구간·제한속도·차로변경 표식·정지선·횡단보도).
- Referee: 행마다 (지도 판정) -> 판정기들 -> Sheet 반영. 판정기는 미래 행을 보지 않고, 필요하면 늦게 낸다.
"""
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.referee.oracle import make_lc_at, make_lim_at, match_inputs
from vtd_rl.referee.sections import FastSections

sf = rs.score_fma


@dataclass
class Hit:
    t: float
    sec: int
    item: int
    level: str
    why: str = ""


class SpanTracker:
    def __init__(self, pred, min_sec):
        self.pred, self.min_sec = pred, min_sec
        self._rows = []

    def _close(self):
        rows, self._rows = self._rows, []
        if rows and rows[-1]["t"] - rows[0]["t"] >= self.min_sec - 1e-9:
            return [(rows[0]["t"], rows[-1]["t"], rows[0], rows)]
        return []

    def update(self, row):
        if self.pred(row):
            self._rows.append(row)
            return []
        return self._close()

    def finish(self):
        return self._close()


@dataclass
class Context:
    board: object
    sections: int
    secs: object
    lim_at: object
    lc_at: object
    tl_stops: dict
    cws: list
    notes: list = field(default_factory=list)

    @classmethod
    def build(cls, board, sections=5):
        m = match_inputs(board)
        secs = FastSections(m.route, sections)
        return cls(board, sections, secs, make_lim_at(secs, m.lims), make_lc_at(secs, m.lc_flag),
                   m.tl_stops, m.cws)


def default_judges(use_map):
    from vtd_rl.referee.judges.contact import ContactJudge, CrosswalkStopJudge
    from vtd_rl.referee.judges.pedestrian import PedestrianJudge
    from vtd_rl.referee.judges.speed import SpeedJudge
    from vtd_rl.referee.judges.traffic_light import TrafficLightJudge
    return [SpeedJudge, TrafficLightJudge, ContactJudge, PedestrianJudge, CrosswalkStopJudge]


class Referee:
    def __init__(self, board, sections=5, use_map=True, judges=None):
        self.ctx = Context.build(board, sections)
        self.map = rs.load_map() if use_map else None
        classes = default_judges(use_map) if judges is None else judges
        self.judges = [cls(self.ctx) for cls in classes]
        self.sheet = sf.Sheet(self.ctx.secs.n)
        self.hits = []
        self.respawns = {}

    def _apply(self, hits):
        for h in hits:
            if h.item == 15:
                self.respawns.setdefault(h.sec, []).append(h.t)
            else:
                self.sheet.hit(h.sec, h.item, h.level, h.why)
            self.hits.append(h)
        return hits

    def step(self, row):
        if self.map is not None:
            rs.check_lanes.classify([row], self.map)
        out = []
        for j in self.judges:
            out += j.step(row)
        return self._apply(out)

    def finish(self):
        out = []
        for j in self.judges:
            out += j.finish()
        return self._apply(out)
