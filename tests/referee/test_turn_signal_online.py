import collections
import copy
import random

from vtd_rl.referee.core import Referee
from vtd_rl.referee.judges.turn_signal import RespawnJudge, TurnSignalJudge
from vtd_rl.referee.oracle import score_episode
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
BOARD = slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def synthetic(rng, n=700):
    """경로를 따라 가며 d_ego 가 가끔 2~4 m 씩 움직이고 지시등이 제멋대로 켜지는 행."""
    rows, s, d, target, sig = [], 0.0, 0.0, 0.0, 0
    for k in range(n):
        if rng.random() < 0.01:
            target = d + rng.choice([-1, 1]) * rng.uniform(2.0, 4.0)
        if rng.random() < 0.02:
            sig = rng.choice([0, 1, 2])
        d += max(-0.08, min(0.08, target - d))
        x, y, h = BOARD.route.point_at(s)
        rows.append(dict(t=round(k * 0.05, 2), x=x, y=y, h=h, v=8.0, tl=0, tid=0, reason="",
                         sig=sig, clr=99.9, d_ego=round(d, 2), objs="", cap_by=""))
        s += 0.35
    return rows


def run_referee(rows, judge):
    ref = Referee(BOARD, use_map=False, judges=[judge])
    for r in rows:
        ref.step(copy.deepcopy(r))
    ref.finish()
    return ref


def test_합성_행_지시등_판정_일치():
    rng = random.Random(7)
    fired = 0
    for _ in range(40):
        rows = synthetic(rng)
        want = collections.Counter(h for h in score_episode(rows, BOARD, use_map=False).hits if h[1] == 13)
        got = collections.Counter((h.sec, h.item, h.level) for h in run_referee(rows, TurnSignalJudge).hits)
        assert got == want
        fired += sum(want.values())
    assert fired > 0


def test_판정은_늦어도_8초_안에_낸다():
    rows = synthetic(random.Random(11), n=900)
    ref = Referee(BOARD, use_map=False, judges=[TurnSignalJudge])
    for r in rows:
        for h in ref.step(copy.deepcopy(r)):
            assert r["t"] - h.t <= 8.0 + 1e-9
    ref.finish()
    assert any(h.item == 13 for h in ref.hits)


def test_합성_순간이동은_채점기와_같이_센다():
    rows = synthetic(random.Random(5), n=300)
    for r in rows[100:]:
        r["y"] += 15.0
    for r in rows[200:]:
        r["y"] += 15.0
    want = score_episode(rows, BOARD, use_map=False).respawns
    ref = run_referee(rows, RespawnJudge)
    assert ref.respawns == want and sum(map(len, want.values())) == 2
