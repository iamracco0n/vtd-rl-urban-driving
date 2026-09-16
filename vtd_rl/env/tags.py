"""심판 ⑧ 면책 입력 — 세계의 물체에서 명령 태그(`cap_by`)를 짓는다.

채점기(`score_fma.item_traffic_light`)는 녹색 신호 앞 정차를 면책할 때 그 프레임의
`reason`/`cap_by` 가 안전 대응 사유(`OBJECT_SAFE`)인지 보고, **그 사유를 뒷받침하는
객체가 실제로 그 자리에 있는지**를 행의 `objs` 로 다시 확인한다
(`_safe_object_cause` -> `_object_backs_reason`). 그래서 여기서 붙이는 태그는
**채점기가 어차피 면책할 정차만** 면책으로 만든다 — 태그를 붙였다고 없는 면책이 생기지 않고,
자리에 맞는 객체가 없으면 태그가 있어도 그대로 감점된다.

태그를 아예 안 붙이면(`cap_by="RL"`) 반대로 **있어야 할 면책이 사라진다**: 녹색에 앞차 뒤로
줄을 서거나 보행자에게 양보한 정차가 ⑧(10 초 넘으면 중대)으로 찍힌다. 단계 ① 판은 교통이 없어
드러나지 않지만, 교통이 있는 M3 판에서는 매번 걸리고 DAgger 가 잘못 붙은 감점을 배운다.

문턱은 새로 지어낸 값이 아니라 채점기 상수 그대로다(`PED_L`·`PED_W`·`PED_MIN_H`·`PED_FAR`·
`PED_LAT`·`AHEAD_FAR`·`AHEAD_LANE_HALF`). 두 값을 따로 적어 두면 어긋나므로 판정식 자체도
채점기의 `_object_backs_reason` 을 그대로 부른다 — 심판이 `_safe_object_cause` 를 부르는 것과 같다.

`reason` 은 "RL" 로 둔다. `LAWFUL_WAIT`(좌회전 화살표 대기 등)는 **스택 자신의 의도**에 대한
것이라 세계에서 지어낼 수 있는 값이 아니다. 규칙 스택이 운전할 때는 스택이 내는 진짜
`reason`/`cap_by` 를 그대로 넘긴다(`TeacherPolicy`) — 이 함수는 학생(RL)이 운전할 때만 쓴다.
"""
import math

from vtd_rl import rule_stack as rs

sf = rs.score_fma

RL = "RL"
# 우선순위 — 사람이 먼저다. 사람은 `YIELD_PED` 로만 근거가 되고(`_object_backs_reason` 은
# `AHEAD_SAFE` 분기에서 사람 치수 객체를 근거로 쓰지 않는다), 차량·장애물은 `FOLLOW` 로만 된다.
CANDIDATES = ("YIELD_PED", "FOLLOW")


def object_row(state, o) -> dict:
    """`state.objects` 의 한 물체를 채점기 `parse_objs_detail` 이 내는 사전 모양으로.

    `dpath` 는 None 이다 — 환경의 `Command` 에는 `rf_objs` 가 없어 로그의 `dpath` 칸이 NaN 이고,
    채점기도 그때 `fy` 로 되돌아간다(`_object_backs_reason` 의 AHEAD 분기). 같은 값을 본다.
    """
    dx, dy = o.x - state.x, o.y - state.y
    ch, sh = math.cos(-state.heading), math.sin(-state.heading)
    return {"fx": dx * ch - dy * sh, "fy": dx * sh + dy * ch,
            "len": o.length, "wid": o.width, "hgt": o.height, "spd": o.speed,
            "clr": 99.9, "dpath": None, "route_w": None, "dh": None}


def world_cap_by(state) -> str:
    """지금 프레임의 물체가 뒷받침하는 안전 대응 태그 — 없으면 "RL"."""
    objs = [object_row(state, o) for o in state.objects]
    for tag in CANDIDATES:
        if any(sf._object_backs_reason(tag, o) for o in objs):
            return tag
    return RL
