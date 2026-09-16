import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.world import dynamics as dyn
from vtd_rl.world.board import Board, load_board
from vtd_rl.world.signals import RouteSignal
from vtd_rl.world.world import World, WorldConfig

LANE = {"lane": -1, "w": 3.5, "l": 1.75, "r": 1.75, "pl": 1.75, "pr": 1.75, "xl": 0.0, "xr": 0.0,
        "need": 0.0, "sig": 0, "j": 0, "jx": 0, "lim": 13.9}

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def straight_board(length=200, actors=(), duration=60.0, signals="always_green"):
    route = [[float(i), 0.0] for i in range(length + 1)]
    sc = rs.Scenario(name="straight", ego_start=[0.0, 0.0, 0.0], ego_goal=route[-1], speed_limit=13.9,
                     duration=duration, actors=list(actors), lights=[], zones=[], ego_route=route,
                     respawns=[], tl_stops={})
    return Board("straight", sc, [dict(LANE) for _ in route], signals)


def drive(world, steer, accel, seconds):
    info = None
    for _ in range(int(round(seconds / world.cfg.dt))):
        _, info = world.step(steer, accel, 0)
        if info.done:
            break
    return info


def test_리셋():
    w = World(straight_board(), signals=[])
    s = w.reset()
    assert (s.x, s.y, s.heading) == (0.0, 0.0, 0.0)
    assert s.speed == 0.0 and (s.tl_id, s.tl_state) == (0, rs.TL_UNSET)   # VTD 는 '신호 없음'을 0 으로 준다
    assert s.t == pytest.approx(1.0e9) and w.t == 0.0


def test_가속하면_경로를_따라_나아간다():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 3.0)
    assert info.outcome == "running"
    assert 4.0 < info.s < 9.0 and info.v > 3.0
    assert info.t == pytest.approx(3.0)


def test_목표():
    w = World(straight_board(length=40), signals=[])
    w.reset()
    info = drive(w, 0.0, 2.0, 20.0)
    assert info.outcome == "goal" and info.done
    assert info.s >= 40.0 - 10.0


def test_도로_이탈():
    w = World(straight_board(), signals=[])
    w.reset()
    info = drive(w, 0.6, 2.0, 20.0)
    assert info.outcome == "offroad"
    assert info.lateral > 1.75 + 1.5


def test_도로_이탈_폭은_합법과_물리_중_넓은_쪽():
    # 규칙 스택 _phys_room 과 같이 그쪽 여유 = max(합법 l/r, 물리 pl/pr)
    b = straight_board()
    for p in b.lane_plan:
        p.update(l=5.0, pl=1.0, r=4.0, pr=2.0)
    b.lane_plan[30] = dict(b.lane_plan[30], l=None, pl=0.0, r=None, pr=None)   # 둘 다 없으면 1.75

    def outcome_at(x, y):
        w = World(b, signals=[])
        w.reset()
        w.ego = dyn.EgoState(x, y, 0.0)
        return w.step(0.0, 0.0, 0)[1].outcome

    assert outcome_at(20.0, 6.0) == "running"         # 왼쪽 5.0 + 1.5 = 6.5 안 (pl 1.0 로 보면 이탈)
    assert outcome_at(20.0, 6.8) == "offroad"
    assert outcome_at(20.0, -5.0) == "running"        # 오른쪽 4.0 + 1.5 = 5.5 안 (pr 2.0 로 보면 이탈)
    assert outcome_at(20.0, -5.8) == "offroad"
    assert outcome_at(30.0, 3.0) == "running"         # 1.75 + 1.5 = 3.25
    assert outcome_at(30.0, -3.5) == "offroad"


def test_시간_초과():
    w = World(straight_board(duration=1.0), signals=[])
    w.reset()
    assert drive(w, 0.0, 0.0, 5.0).outcome == "timeout"


def test_정체():
    cfg = WorldConfig(stall_seconds=2.0)
    w = World(straight_board(), config=cfg, signals=[])
    w.reset()
    info = drive(w, 0.0, 0.0, 5.0)
    assert info.outcome == "stalled" and info.t == pytest.approx(2.0, abs=0.06)


def test_신호_알림():
    w = World(straight_board(), signals=[RouteSignal(50.0, 7)])
    s = w.reset()
    assert (s.tl_id, s.tl_state) == (7, rs.TL_GREEN)


def test_물체는_80m_안만():
    near = rs.Actor(id=1, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                    motion={"kind": "static", "pos": [40.0, 3.0]})
    far = rs.Actor(id=2, type="object", size=[1.0, 1.0, 1.0], spawn={"at_time": 0},
                   motion={"kind": "static", "pos": [150.0, 0.0]})
    w = World(straight_board(actors=[near, far]), signals=[])
    s = w.reset()
    assert [o.id for o in s.objects] == [1]


def test_리셋하면_처음부터():
    w = World(straight_board(), signals=[])
    w.reset()
    drive(w, 0.0, 2.0, 2.0)
    s = w.reset()
    assert (s.x, w.t, w.ego.v) == (0.0, 0.0, 0.0)


def test_리셋_시드():
    b = load_board(H, signals="cycle")
    w = World(b, seed=1)
    w.reset(7)
    first = [w.reporter.report(0.0, 0, t)[1] for t in range(0, 60, 5)]
    w.reset(7)
    assert [w.reporter.report(0.0, 0, t)[1] for t in range(0, 60, 5)] == first
    assert w.seed == 7
    w.reset(8)
    assert w.seed == 8
    w.reset()
    assert w.seed == 8                      # 시드를 안 주면 마지막 시드를 그대로 쓴다


def test_상태_속도는_세계가_채운다():
    w = World(load_board(H))
    s = w.reset()
    assert s.speed == 0.0
    for _ in range(20):
        s, info = w.step(0.0, 1.0, 0)
    assert s.speed == pytest.approx(w.ego.v) and s.speed > 0.5
    assert s.speed_raw == pytest.approx(w.ego.v)
