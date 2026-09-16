"""명령 태그(reason·cap_by) — 채점기 ⑧ 면책이 환경 안에서도 살아 있는가."""
import csv

import numpy as np
import pytest

from vtd_rl import rule_stack as rs
from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.tags import world_cap_by
from vtd_rl.env.teacher_policy import run_teacher_in_env
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
sf = rs.score_fma

STOP_S = 72.0        # 여기서 급제동 -> 신호(s≈96.3) 앞 12 m 쯤에 선다(⑧ 판정 구간 30 m 안)
OBJ_S = 115.0        # 정지한 자차에서 30 m 앞 — AHEAD_FAR(60 m) 안, 접촉은 안 난다
HOLD_STEPS = 130     # 13 초 정차 > GREEN_MAJOR(10 초)


def green_board(name, obj_s=None):
    """녹색 신호 하나가 있는 코스 H 앞 250 m. obj_s 를 주면 그 자리에 정지 차량 하나."""
    b = slice_board(load_board(H), 0.0, 250.0, name, signals="always_green")
    if obj_s is not None:
        x, y, _h = b.route.point_at(obj_s)
        b.scenario.actors = [rs.Actor(id=901, type="vehicle", size=[4.5, 1.8, 1.5],
                                      spawn={"at_time": 0.0}, motion={"kind": "static", "pos": [x, y]})]
    return b


def green_idle_items(board, log_dir=None):
    """녹색 신호 앞에서 13 초 섰다가 다시 출발 — 그동안 나온 심판 항목."""
    env = VtdDriveEnv([board], EnvConfig(log_dir=log_dir))
    env.reset(seed=0)
    items = []

    def go(control, n, until_s=None):
        nonlocal items
        for _ in range(n):
            _o, _r, term, trunc, info = env.step({"control": np.array(control, np.float32), "turn": 0})
            items += [(h[2], h[3]) for h in info["hits"]]
            if term or trunc or (until_s is not None and info["s"] >= until_s):
                return True
        return False

    assert go([0.0, 0.35], 600, until_s=STOP_S)          # 신호 앞까지 간다
    assert not go([0.0, -1.0], HOLD_STEPS)               # 13 초 정차
    assert not go([0.0, 0.5], 20)                        # 다시 출발 -> 정차 구간이 닫힌다
    env.close()
    return items


def test_녹색_무의미_정차는_앞차가_있으면_8번이_아니다():
    """같은 정차인데 앞차가 있으면 면책, 없으면 ⑧ 중대 — 태그가 채점기와 같은 답을 낸다.

    태그를 안 붙이면(예전처럼 cap_by="RL") **두 경우 다** ⑧ 이 붙는다. 단계 ① 판은 교통이 없어
    드러나지 않지만 M3 판에서는 줄 서서 기다릴 때마다 걸린다.
    """
    assert (8, "major") in green_idle_items(green_board("H_green_alone"))
    assert not any(i == 8 for i, _lv in green_idle_items(green_board("H_green_follow", OBJ_S)))


def test_선생님의_진짜_reason_이_행에_남는다(tmp_path):
    """규칙 스택이 운전하면 행의 reason·cap_by 는 스택이 낸 값 그대로다("RL" 로 덮지 않는다)."""
    b = slice_board(load_board(H), 0.0, 250.0, "H_0_250_tags")
    env = VtdDriveEnv([b], EnvConfig(log_dir=str(tmp_path)))
    out = run_teacher_in_env(env, seed=0)
    env.close()
    with open(out["info"]["result"]["rows_csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert any(r["reason"] not in ("RL", "") for r in rows), {r["reason"] for r in rows}
    assert any(r["cap_by"] not in ("RL", "") for r in rows), {r["cap_by"] for r in rows}


class _State:
    def __init__(self, objects):
        self.x = self.y = self.heading = 0.0
        self.objects = objects


def _obj(x, y, length, width, height, speed=0.0):
    return rs.Obj(id=1, x=x, y=y, z=0.0, heading=0.0, speed=speed,
                  length=length, width=width, height=height)


@pytest.mark.parametrize("name,objs,want", [
    ("빈 세계", [], "RL"),
    ("앞차", [_obj(10.0, 0.0, 4.5, 1.8, 1.5)], "FOLLOW"),
    ("보행자", [_obj(10.0, 4.0, 0.6, 0.7, 1.8)], "YIELD_PED"),
    ("사람이 먼저", [_obj(10.0, 0.0, 4.5, 1.8, 1.5), _obj(5.0, 3.0, 0.6, 0.7, 1.8)], "YIELD_PED"),
    ("옆 차로 차", [_obj(10.0, 5.0, 4.5, 1.8, 1.5)], "RL"),          # |fy| > AHEAD_LANE_HALF
    ("너무 먼 차", [_obj(80.0, 0.0, 4.5, 1.8, 1.5)], "RL"),          # fx > AHEAD_FAR
    ("뒤차", [_obj(-20.0, 0.0, 4.5, 1.8, 1.5)], "RL"),
    ("노면물", [_obj(10.0, 0.0, 1.0, 1.0, 0.1)], "RL"),              # hgt < SAFE_OBJ_MIN_H
])
def test_세계에서_지은_태그(name, objs, want):
    assert world_cap_by(_State(objs)) == want, name
