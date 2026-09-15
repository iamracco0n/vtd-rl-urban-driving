"""⑪ 장애물 충돌 · ⑭ 차량·보행자 접촉 · ⑫ 횡단보도 위 정지 — score_fma.item_contact, item_crosswalk_stop.

⑪⑭ 은 그 프레임에, ⑫ 는 정지가 CW_STOP_S 에 닿는 프레임에 낸다(등급이 늘 경미라 정지가 끝나기를 기다리지 않는다).
"""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit, SpanTracker

sf = rs.score_fma
PERSON_MIN_H = 1.2      # item_contact 안의 숫자 — 사람·이륜차 최저 키[m]
CW_RADIUS = 3.0         # item_crosswalk_stop 안의 숫자 — 횡단보도 중심 반경[m]


class ContactJudge:
    items = (11, 14)

    def __init__(self, ctx):
        self.ctx = ctx

    def step(self, r):
        if r["clr"] >= 0.0:
            return []
        sec = self.ctx.secs.of(r["x"], r["y"])
        near = min(sf.parse_objs(r["objs"]), key=lambda o: o[6], default=None)
        if near is None:
            return [Hit(r["t"], sec, 11, "minor", f"t={r['t']:.1f} 여유 {r['clr']:.2f}m")]
        _fx, _fy, ol, ow, oh, _sp, _clr = near
        person = ol <= sf.PED_L and ow <= sf.PED_W and oh >= PERSON_MIN_H
        vehicle = ol > sf.PED_L
        item = 14 if (person or vehicle) else 11
        return [Hit(r["t"], sec, item, "major" if item == 14 else "minor",
                    f"t={r['t']:.1f} 여유 {r['clr']:.2f}m (상대 {ol:.1f}x{ow:.1f}m)")]

    def finish(self):
        return []


class CrosswalkStopJudge:
    items = (12,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.span = SpanTracker(self._on_crosswalk, sf.CW_STOP_S, emit="reach")

    def _on_crosswalk(self, r):
        fx = r["x"] + sf.FRONT / 2 * math.cos(r["h"])     # 채점기가 쓰는 '차체 중앙쯤'
        fy = r["y"] + sf.FRONT / 2 * math.sin(r["h"])
        return (r["v"] <= sf.STOP_V
                and any(math.hypot(fx - c["x"], fy - c["y"]) < CW_RADIUS for c in self.ctx.cws))

    def _hits(self, spans):
        return [Hit(t0, self.ctx.secs.of(r0["x"], r0["y"]), 12, "minor",
                    f"t={t0:.1f}~{t1:.1f} 횡단보도 위 {t1-t0:.1f}초째 정지")
                for t0, t1, r0, _ in spans]

    def step(self, r):
        return self._hits(self.span.update(r))

    def finish(self):
        return self._hits(self.span.finish())
