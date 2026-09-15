"""⑩ 보행자 대응 — score_fma.item_pedestrian 의 한 행 판.

채점기는 후보 프레임 t 에서 [t-6, t+1] 초 안에 한 번이라도 섰는지 본다. 그래서 1 초 뒤에 판정한다.
한 프레임에 사람이 여럿이어도 '섰나'는 시각만 보므로 프레임당 감점은 많아야 한 번이다.
"""
from collections import deque

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma
PED_STOP_WIN = 6.0      # item_pedestrian 안의 지역 상수 — 이 시간 안에 섰으면 '섰다'[s]
PED_AFTER = 1.0         # item_pedestrian.stopped_near 의 뒤쪽 창[s]


class PedestrianJudge:
    items = (10,)

    def __init__(self, ctx):
        self.ctx = ctx
        self._stops = deque()      # STOP_V 이하였던 행의 t
        self._pending = deque()    # (후보 행, 첫 사람의 fy)

    def _person_ahead(self, r):
        if r["v"] <= sf.STOP_V:
            return None
        for fx, fy, ol, ow, _oh, sp, _clr in sf.parse_objs(r["objs"]):
            if not (ol <= sf.PED_L and ow <= sf.PED_W):
                continue
            if not (0.0 <= fx <= sf.FRONT + sf.PED_NEAR):
                continue
            if not (0.3 <= sp <= 2.5):
                continue
            if abs(fy) <= sf.PED_NEAR:
                return fy
        return None

    def _judge(self, r, fy):
        t_at = r["t"]
        if any(t_at - PED_STOP_WIN <= s <= t_at + PED_AFTER for s in self._stops):
            return []
        return [Hit(t_at, self.ctx.secs.of(r["x"], r["y"]), 10, "major",
                    f"t={t_at:.1f} 보행자 옆 {abs(fy):.1f}m 를 {r['v']*3.6:.0f}km/h 로 안 서고 통과")]

    def step(self, r):
        if r["v"] <= sf.STOP_V:
            self._stops.append(r["t"])
        out = []
        while self._pending and r["t"] > self._pending[0][0]["t"] + PED_AFTER:
            out += self._judge(*self._pending.popleft())
        fy = self._person_ahead(r)
        if fy is not None:
            self._pending.append((r, fy))
        oldest = self._pending[0][0]["t"] if self._pending else r["t"]
        while self._stops and self._stops[0] < oldest - PED_STOP_WIN:
            self._stops.popleft()
        return out

    def finish(self):
        out = []
        while self._pending:
            out += self._judge(*self._pending.popleft())
        return out
