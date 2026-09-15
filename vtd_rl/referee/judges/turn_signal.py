"""⑬ 차로변경 지시등 — score_fma.item_turn_signal 의 한 행 판(최대 8 초 늦게 낸다).
⑮ 리스폰 — score_fma.item_respawn 의 한 행 판(오프라인 세계에는 순간이동이 없어 합성 행으로 검증한다)."""
import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Hit

sf = rs.score_fma
LANE_W, WIN = 3.0, 8.0      # item_turn_signal 안의 지역 상수
ONSET_D = 0.3               # item_turn_signal 안의 숫자 — 기동 개시로 볼 횡이동[m]
SPAWN_M = 8.0               # item_turn_signal 안의 숫자 — 출발점에서 이만큼은 뺀다[m]
RESPAWN_DT = 1.0            # item_respawn 안의 숫자 — 이 시간 안의 순간이동만[s]


class TurnSignalJudge:
    items = (13,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.start = None
        self.rows = []              # (행, 같은 sig 가 이어진 첫 행의 t) — 앵커부터
        self.i = 0                  # 앵커(버퍼 안 번호)
        self.j = 0                  # 앵커 창에서 마지막으로 본 행
        self.moved = 0.0
        self.last_sig = None
        self.last_run_start = None

    def step(self, r):
        if self.start is None:
            self.start = (r["x"], r["y"])
        if math.hypot(r["x"] - self.start[0], r["y"] - self.start[1]) <= SPAWN_M:
            return []
        if r["sig"] != self.last_sig:
            self.last_sig, self.last_run_start = r["sig"], r["t"]
        self.rows.append((r, self.last_run_start))
        return self._advance(final=False)

    def finish(self):
        return self._advance(final=True)

    def _advance(self, final):
        out, rows = [], self.rows
        while self.i < len(rows):
            r0 = rows[self.i][0]
            while (abs(self.moved) < LANE_W and self.j + 1 < len(rows)
                   and rows[self.j + 1][0]["t"] - r0["t"] < WIN):
                self.j += 1
                self.moved = rows[self.j][0]["d_ego"] - r0["d_ego"]
            if abs(self.moved) >= LANE_W:
                out += self._judge(self.i, self.j, self.moved)
                self.i = self.j + 1
            elif self.j + 1 < len(rows) or final:
                self.i += 1
            else:
                break                                   # 창이 아직 열려 있다 — 다음 행을 기다린다
            self.j, self.moved = self.i, 0.0
        del rows[:self.i]
        self.j -= self.i
        self.i = 0
        return out

    def _judge(self, i, j, moved):
        rows = self.rows
        d0 = rows[i][0]["d_ego"]
        k0 = next((k for k in range(i, j + 1) if abs(rows[k][0]["d_ego"] - d0) > ONSET_D), i)
        r0, run_start = rows[k0]
        want = 1 if moved > 0 else 2                    # TS_LEFT / TS_RIGHT
        lead = r0["t"] - run_start if r0["sig"] == want else None
        if lead is not None and lead >= sf.SIG_SEC:
            return []
        return [Hit(r0["t"], self.ctx.secs.of(r0["x"], r0["y"]), 13, "minor",
                    f"t={r0['t']:.1f} 차로변경 선행점등 {0.0 if lead is None else lead:.1f}초 < {sf.SIG_SEC}초")]


class RespawnJudge:
    items = (15,)

    def __init__(self, ctx):
        self.ctx = ctx
        self.prev = None

    def step(self, r):
        a, self.prev = self.prev, r
        if a is None:
            return []
        d = math.hypot(r["x"] - a["x"], r["y"] - a["y"])
        if d > sf.RESPAWN_JUMP and r["t"] - a["t"] < RESPAWN_DT:
            return [Hit(r["t"], self.ctx.secs.of(r["x"], r["y"]), 15, "major",
                        f"t={r['t']:.1f} {d:.0f}m 순간이동")]
        return []

    def finish(self):
        return []
