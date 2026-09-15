"""지도 없는 심판 전체(Referee use_map=False) 대 score_fma — 무작위 합성 행으로 빠르게.

대본 판·선생님 코스는 판마다 한두 갈래만 지난다. 여기서는 코스 H 0~250 m 판 위에 행을 직접 만들어
한 판에 여러 갈래를 섞는다.
- 신호151 정지선 30 m 안에 서기: 녹색이면 ⑧(경미·중대), 적색·점멸이면 멀리 서서 ⑦⑨ 경미,
  정지선 바로 앞 [0, 2) m 에 서면 면책.
- 정차 중 reason·cap_by 와 근거 객체(과반·소수), LAWFUL_WAIT 사유 — ⑧ 의 두 면책.
- 횡단보도 위 정지(⑫), 앞을 지나는 사람(⑩), 접촉(⑪⑭), 한 프레임 순간이동(⑮).
순간이동은 정지선과 횡단보도를 다 지난 JUMP_FROM 뒤에서만 한다. 정지선 앞에 서 있는 동안 튀면
⑦⑨ 판정 시점을 일부러 다르게 둔 탓에 답이 갈린다(judges/traffic_light.py docstring).
"""
import collections
import math
import random
from unittest import mock

from vtd_rl import rule_stack as rs
from vtd_rl.referee.core import Referee
from vtd_rl.referee.oracle import match_inputs, score_episode
from vtd_rl.world.board import load_board, slice_board

sf = rs.score_fma
H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
BOARD = slice_board(load_board(H), 0.0, 250.0, "H_0_250")
DT = 0.05
EPISODES = 30
LINE_S = 96.3                      # 신호151 정지선 DB 점의 경로 s(scenarios.H_TL151)
CW_S = 130.7 - sf.FRONT / 2        # 차체 중앙쯤이 신호 횡단보도 중심(경로 s=130.7)에 오는 뒷축 s
JUMP_FROM = 150.0
END_S = 200.0
OTHER = ("", "CRUISE", "OVT:PASS", "NARROW_BLOCK", "YIELD_ONCOMING", "LC_REAR_YIELD")   # 근거 객체 없이는 면책 못 받는 사유
VEH = "{fx:.1f}:{fy:.1f}:4.5:1.8:1.5:0.0:{clr:.2f}:0.30:nan:0"
PED = "{fx:.1f}:{fy:.1f}:0.6:0.7:{h:.1f}:{sp:.1f}:{clr:.2f}:nan:nan:0"
CONE = "{fx:.1f}:{fy:.1f}:0.2:0.5:0.6:0.0:{clr:.2f}:nan:nan:0"


def _stop_plan(rng):
    """정차 일정 — (뒷축 s, 프레임 수, 정차 중 사유·객체 방식)."""
    at = []
    if rng.random() < 0.8:
        at.append(rng.uniform(64.0, 76.0))                              # 정지선 20~32 m 앞
    if rng.random() < 0.8:
        at.append(rng.uniform(80.0, 90.0) if rng.random() < 0.7
                  else LINE_S - sf.FRONT - rng.uniform(0.2, 1.8))       # 앞범퍼가 [0, 2) m
    if rng.random() < 0.5:
        at.append(CW_S + rng.uniform(-1.0, 1.0))
    return [dict(s=s, n=rng.choice((rng.randint(12, 60), rng.randint(105, 190), rng.randint(205, 260))),
                 mode=rng.choice(("none", "none", "weak_obj", "obj", "obj_cap", "lawful", "ped")))
            for s in at]


def _stop_cues(rng, mode):
    """정차 중 한 프레임의 (reason, cap_by, 객체)."""
    reason, cap_by, objs = rng.choice(OTHER), rng.choice(OTHER), []
    if mode in ("obj", "obj_cap", "weak_obj"):
        if mode == "obj_cap":
            cap_by = "FOLLOW"                                           # reason 이 덮여도 cap_by 로 면책
        else:
            reason = "FOLLOW"
        if rng.random() < (0.3 if mode == "weak_obj" else 0.9):         # 소수면 면책 안 됨
            objs.append(VEH.format(fx=8.0, fy=0.2, clr=3.0))
    elif mode == "ped":
        reason = "YIELD_PED"
        if rng.random() < 0.8:
            objs.append(PED.format(fx=9.0, fy=2.0, h=1.7, sp=1.3, clr=5.0))
    elif mode == "lawful" and rng.random() < 0.85:
        reason = rng.choice(sf.LAWFUL_WAIT)
    return reason, cap_by, objs


def synthetic(rng):
    tl = rng.choice((1, 3, 3, 4, 5, 6))
    tid = rng.choice((151,) * 8 + (0, -1))
    plan = _stop_plan(rng)
    lat = rng.choice((0.0, 1.0))
    rows, s, v, k = [], rng.uniform(40.0, 60.0), rng.uniform(6.0, 12.0), 0
    stop, left = None, 0
    while s < END_S:
        if rng.random() < 0.001:
            tl = rng.choice((1, 3, 4, 5, 6))
        if left == 0 and plan and s >= plan[0]["s"]:
            stop = plan.pop(0)
            left = stop["n"]
        if left > 0:
            v = 0.6 if rng.random() < 0.002 else rng.choice((0.0, 0.0, 0.0, 0.2))   # 가끔 정차가 끊긴다
            left -= 1
        else:
            stop = None
            v = max(4.0, min(21.0, v + rng.uniform(-0.8, 0.8)))
        x, y, h = BOARD.route.point_at(s)
        x, y = x - lat * math.sin(h), y + lat * math.cos(h)
        if s > JUMP_FROM and rng.random() < 0.003:                       # 한 프레임 옆으로 튄다(리스폰 2 번)
            x, y = x - 12.0 * math.sin(h), y + 12.0 * math.cos(h)
        if stop is not None:
            reason, cap_by, objs = _stop_cues(rng, stop["mode"])
        else:
            reason, cap_by, objs = rng.choice(OTHER), rng.choice(OTHER), []
        if rng.random() < 0.02:                                          # 앞을 지나는 사람·빠른 이륜차
            objs.append(PED.format(fx=rng.uniform(0.0, 8.0), fy=rng.uniform(-4.0, 4.0), h=rng.choice((1.7, 1.0)),
                                   sp=rng.choice((0.0, 0.5, 1.3, 3.0)), clr=2.0))
        clr = 99.9
        if rng.random() < 0.01:                                          # 접촉 — 가장 가까운 객체가 ⑪·⑭ 를 가른다
            clr = -0.1
            kind = rng.choice(("none", "cone", "ped", "short", "veh"))
            if kind == "cone":
                objs.append(CONE.format(fx=2.0, fy=0.5, clr=clr))
            elif kind == "veh":
                objs.append(VEH.format(fx=4.0, fy=0.5, clr=clr))
            elif kind != "none":
                objs.append(PED.format(fx=2.0, fy=0.5, h=1.7 if kind == "ped" else 0.9, sp=0.0, clr=clr))
        rows.append(dict(t=round(k * DT, 2), x=round(x, 3), y=round(y, 3), h=round(h, 4), v=round(v, 2),
                         tl=tl, tid=tid, reason=reason, sig=0, clr=clr, d_ego=lat,
                         objs="|".join(objs), cap_by=cap_by))
        s += v * DT
        k += 1
    return rows


class _Count8(sf.Sheet):
    def __init__(self, n):
        super().__init__(n)
        self.n8 = 0

    def hit(self, sec, item, level, why):
        self.n8 += item == 8
        super().hit(sec, item, level, why)


def _idle_verdicts(rows, m, secs):
    """채점기가 ⑧ 정차 구간에 내린 감점·'판정 불명' 메모 수(객체 면책으로 빠진 구간은 안 센다)."""
    sheet, notes = _Count8(secs.n), []
    sf.item_traffic_light(rows, sheet, secs, m.tl_stops, notes)
    return sheet.n8 + len(notes)


def test_무작위_합성_행_지도_없는_전_항목_일치():
    m = match_inputs(BOARD)
    secs = sf.Sections(m.route, 5)
    fired = collections.Counter()
    notes = respawns = obj_exempt = 0
    for seed in range(EPISODES):
        rows = synthetic(random.Random(seed))
        want = score_episode(rows, BOARD, use_map=False)
        ref = Referee(BOARD, use_map=False)
        for r in rows:
            ref.step(dict(r))
        ref.finish()
        got = collections.Counter((h.sec, h.item, h.level) for h in ref.hits if h.item != 15)
        want_hits = collections.Counter(want.hits)
        assert got == want_hits, f"seed={seed} 채점기에만 {dict(want_hits - got)} 심판에만 {dict(got - want_hits)}"
        assert ref.respawns == want.respawns, f"seed={seed} 리스폰 채점기 {want.respawns} 심판 {ref.respawns}"
        assert len(ref.ctx.notes) == len(want.notes), f"seed={seed} 메모 채점기 {want.notes} 심판 {ref.ctx.notes}"
        fired.update({(item, level): n for (_sec, item, level), n in want_hits.items()})
        notes += len(ref.ctx.notes)
        respawns += sum(map(len, want.respawns.values()))
        with mock.patch.object(sf, "_safe_object_cause", lambda r: False):
            no_obj = _idle_verdicts(rows, m, secs)
        obj_exempt += no_obj - _idle_verdicts(rows, m, secs)

    # 이 판들이 실제로 각 갈래를 지났나 — 안 지났으면 위의 일치는 그 갈래를 검증하지 못한다.
    assert fired[(8, "minor")] > 0 and fired[(8, "major")] > 0, fired
    assert obj_exempt > 0, "⑧ 객체 근거 과반 면책이 한 번도 없었다"
    assert notes > 0, "⑧ LAWFUL_WAIT '판정 불명' 메모가 한 번도 없었다"
    assert fired[(7, "minor")] + fired[(9, "minor")] > 0, fired
    for key in ((10, "major"), (11, "minor"), (12, "minor"), (14, "major")):
        assert fired[key] > 0, (key, fired)
    assert respawns > 0
