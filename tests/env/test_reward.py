import pytest

from vtd_rl.env.board_index import board_index
from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper, ViolationTracker
from vtd_rl.referee.core import Hit
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_board():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_같은_항목은_1초에_한_번만_센다():
    tr = ViolationTracker(1.0)
    hits = [Hit(t=k * 0.05, sec=0, item=1, level="minor") for k in range(60)]   # 3 초 연속 과속
    counted = [h for k in range(60) for h in tr.count([hits[k]])]
    assert [round(h.t, 2) for h in counted] == [0.0, 1.0, 2.0]


def test_항목이_다르면_따로_센다():
    tr = ViolationTracker(1.0)
    counted = tr.count([Hit(0.0, 0, 1, "minor"), Hit(0.0, 0, 3, "minor"), Hit(0.05, 0, 1, "minor")])
    assert [(h.item, h.t) for h in counted] == [(1, 0.0), (3, 0.0)]


def test_진행과_완주와_시간():
    b = h_board()
    cfg = RewardConfig()
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([], b.route.total / 10.0, zero, zero, "running")
    assert out.terms["progress"] == pytest.approx(cfg.progress_total / 10.0)
    assert out.terms["time"] == pytest.approx(cfg.time_cost * -1.0)
    assert out.terms["goal"] == 0.0
    done = sh.step([], 0.0, zero, zero, "goal")
    assert done.terms["goal"] == pytest.approx(cfg.goal_bonus)


def test_위반과_충돌과_이탈():
    b = h_board()
    cfg = RewardConfig(rule_scale=2.0)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 1, "minor"), Hit(1.0, 0, 4, "major")], 0.0, zero, zero, "running")
    assert out.terms["violation"] == pytest.approx((cfg.minor + cfg.major) * cfg.rule_scale)
    assert out.collision is False
    hit_out = sh.step([Hit(2.0, 0, 14, "major")], 0.0, zero, zero, "running")
    assert hit_out.terms["collision"] == pytest.approx(cfg.collision)
    assert hit_out.terms["violation"] == 0.0 and hit_out.collision is True and hit_out.counted == 1
    off = sh.step([], 0.0, zero, zero, "offroad")
    assert off.terms["offroad"] == pytest.approx(cfg.offroad)


def test_승차감은_변화량에_붙는다():
    b = h_board()
    cfg = RewardConfig()
    sh = RewardShaper(b, cfg)
    sh.reset()
    prev = {"control": [0.0, 0.0], "turn": 0}
    now = {"control": [0.5, -0.4], "turn": 0}
    out = sh.step([], 0.0, now, prev, "running")
    assert out.terms["comfort"] == pytest.approx(cfg.comfort_steer * 0.5 + cfg.comfort_accel * 0.4)
    same = sh.step([], 0.0, now, now, "running")
    assert same.terms["comfort"] == 0.0


def test_총합은_항의_합이다():
    b = h_board()
    sh = RewardShaper(b)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 2, "major")], 5.0, zero, zero, "running")
    assert out.total == pytest.approx(sum(out.terms.values()))
