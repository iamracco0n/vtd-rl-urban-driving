"""⑦ 적색 정지 · ⑧ 녹색 무의미 정차 · ⑨ 적색점멸 — score_fma.item_traffic_light 의 한 행 판.

⑦⑨ 는 (신호, 상태)마다 선을 처음 넘는 프레임에 확정한다. 채점기는 판 끝에 보므로, 같은 (신호, 상태)로
다시 다가가 서는 드문 경우만 답이 다르다(M2a 계획 Global Constraints).
⑧ 은 정차 구간 전체의 다수결 면책이 있어 구간이 닫힐 때 낸다.
"""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
TL_RED, TL_GREEN, TL_LEFT, TL_GREEN_LEFT, TL_FLASH = 1, 3, 4, 5, 6   # item_traffic_light 안의 지역 상수


def _stop_point(ctx, tid):
    return ctx.tl_stops.get(str(tid)) or ctx.tl_stops.get(tid)


def _forward(p, r):
    dx, dy = p[0] - r["x"], p[1] - r["y"]
    return dx * math.cos(-r["h"]) - dy * math.sin(-r["h"])


class TrafficLightJudge:
    items = (7, 8, 9)

    def __init__(self, ctx):
        self.ctx = ctx
        self.held, self.far_held, self.ahead = {}, {}, {}
        self.stopped_since, self.far_stopped_since = {}, {}
        self.crossed = set()
        self.idle = SpanTracker(self._green_idle, sf.GREEN_MINOR)

    def step(self, r):
        return self._red(r) + self._idle_hits(self.idle.update(r))

    def finish(self):
        return self._idle_hits(self.idle.finish())

    def _red(self, r):
        tid = r["tid"]
        p = _stop_point(self.ctx, tid)
        if tid < 0 or p is None:
            return []
        fwd = _forward(p, r) - sf.FRONT                       # 앞범퍼 기준
        key = (tid, r["tl"])
        if fwd > 0.0:
            self.ahead[key] = True
        if r["tl"] not in (TL_RED, TL_FLASH):
            return []
        if 0.0 <= fwd < sf.STOP_NEAR and r["v"] <= sf.STOP_V:
            self.far_stopped_since.pop(key, None)
            t0 = self.stopped_since.setdefault(key, r["t"])
            if r["t"] - t0 >= sf.STOP_HOLD:
                self.held[key] = True
        elif fwd >= sf.STOP_NEAR and r["v"] <= sf.STOP_V:
            self.stopped_since.pop(key, None)
            t0 = self.far_stopped_since.setdefault(key, r["t"])
            if r["t"] - t0 >= sf.STOP_HOLD:
                self.far_held[key] = True
        else:
            self.stopped_since.pop(key, None)
            self.far_stopped_since.pop(key, None)
        if not (fwd < -1.0 and self.ahead.get(key)) or key in self.crossed:
            return []
        self.crossed.add(key)
        if self.held.get(key):
            return []
        item = 7 if r["tl"] == TL_RED else 9
        sec = self.ctx.secs.of(r["x"], r["y"])
        if self.far_held.get(key):
            return [Hit(r["t"], sec, item, "minor",
                        f"t={r['t']:.1f} 신호{tid} 정지선 {sf.STOP_NEAR}m 이상 앞에 정지")]
        return [Hit(r["t"], sec, item, "major", f"t={r['t']:.1f} 신호{tid} 정지 없이 통과")]

    def _green_idle(self, r):
        p = _stop_point(self.ctx, r["tid"])
        if p is None or r["tl"] not in (TL_GREEN, TL_LEFT, TL_GREEN_LEFT):
            return False
        return 0.0 < _forward(p, r) < sf.GREEN_NEAR and r["v"] <= sf.STOP_V   # 뒷축 기준

    def _idle_hits(self, spans):
        out = []
        for t0, t1, r0, span_rows in spans:
            held_s = t1 - t0
            why = f"t={t0:.1f}~{t1:.1f} 신호{r0['tid']} 녹색에 {held_s:.1f}초 정차"
            obj_ok = sum(1 for r in span_rows if sf._safe_object_cause(r))
            if obj_ok > len(span_rows) * sf.SAFE_MAJORITY:
                continue
            lawful = [r["reason"] for r in span_rows if r["reason"] in sf.LAWFUL_WAIT]
            if len(lawful) > len(span_rows) * sf.SAFE_MAJORITY:
                self.ctx.notes.append(f"[⑧ 판정 불명] {why}")
                continue
            out.append(Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), 8,
                           "major" if held_s >= sf.GREEN_MAJOR else "minor", why))
        return out
