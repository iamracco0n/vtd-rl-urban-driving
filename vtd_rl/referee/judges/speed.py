"""① 제한속도 · ② 보호구역 속도 — score_fma.item_speed 의 한 행 판."""
from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma


class SpeedJudge:
    items = (1, 2)

    def __init__(self, ctx):
        self.ctx = ctx

    def step(self, r):
        lim = self.ctx.lim_at(r["x"], r["y"])
        if lim is None:
            return []
        over = r["v"] - lim
        if lim <= sf.ZONE_LIM:
            item, major, tol = 2, sf.ZONE_MAJOR, sf.ZONE_TOL
        else:
            item, major, tol = 1, sf.SPD_MAJOR, sf.SPD_TOL
        level = "major" if over > major else ("minor" if over > tol else None)
        if level is None:
            return []
        return [Hit(r["t"], self.ctx.secs.of(r["x"], r["y"]), item, level,
                    f"t={r['t']:.1f} {r['v']*3.6:.0f}>{lim*3.6:.0f}km/h")]

    def finish(self):
        return []
