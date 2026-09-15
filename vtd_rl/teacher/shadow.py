"""규칙 스택 그림자 주행.

대회 제어PC(`src/main.py`)와 같은 인자로 DrivingStack 을 만든다. 차이는 둘뿐이다:
- now 에 벽시계 대신 **시뮬 시계**(World.clock)를 넣는다 — 규칙 스택 판단 코드는 벽시계를 직접 읽지 않는다
  (`overtake.py` 는 now 가 None 일 때만 time.time() 을 쓰고, drive 가 now 를 넘긴다. 2026-09-15 확인).
- 속도는 VTDLink 의 위치 미분 추정기를 소켓 없이 그대로 쓴다(VTD 패킷에 자차 속도가 없다).
"""
from vtd_rl import rule_stack as rs


class ShadowTeacher:
    def __init__(self, board):
        db = rs.map_db()
        self.stack = rs.DrivingStack(
            scenario=board.scenario, base_limit=board.scenario.speed_limit,
            tl_stops_extra=db["tl_map"], max_speed=None, lane_plan=board.lane_plan,
            turn_lane=True, crosswalks=db["crosswalks"], stoplines=db["stoplines"],
            stoplines_all=db["stoplines_all"], allow_centerline_escape=False)
        self._speed = rs.VTDLink("127.0.0.1")          # connect() 하지 않는다
        self._last_now = None

    def act(self, state, now):
        self._speed._update_speed(state, now)
        dt = 0.05 if self._last_now is None else max(1e-3, now - self._last_now)
        self._last_now = now
        return self.stack.step(state, dt, now)
