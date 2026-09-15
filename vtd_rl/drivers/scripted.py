"""대본 운전자 — 속도·횡오프셋·지시등을 경로 거리 s 와 시각 t 의 일정으로 준다.

일부러 규칙을 어기는 판을 만들어 심판과 채점기가 같은 판정을 내는지 보는 시험 도구다.
조향은 규칙 스택 PurePursuit, 속도는 LongPI 를 그대로 쓴다.
"""
import math

from vtd_rl import rule_stack as rs

AHEAD_PTS = 40          # 조향 경로로 넘길 앞쪽 경로점 수(점 간격 약 1.4 m)
STOP_REACHED = 0.3      # StopAt: 남은 거리가 이 안이면 도착으로 친다[m]


class Cruise:
    def __init__(self, v):
        self.v = v

    def __call__(self, s, t):
        return self.v


class StopAt:
    def __init__(self, s_stop, hold, v, decel=2.0):
        self.s_stop, self.hold, self.v, self.decel = s_stop, hold, v, decel
        self._since = None
        self._done = False

    def __call__(self, s, t):
        if self._done:
            return self.v
        if self._since is None:
            remain = self.s_stop - s
            if remain > STOP_REACHED:
                return min(self.v, math.sqrt(2.0 * self.decel * remain))
            self._since = t
        if t - self._since < self.hold:
            return 0.0
        self._done = True
        return self.v


class Offset:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, s, t):
        return self.value


class LaneShift:
    def __init__(self, s_start, delta, length):
        self.s_start, self.delta, self.length = s_start, delta, length

    def __call__(self, s, t):
        return self.delta * max(0.0, min(1.0, (s - self.s_start) / self.length))


class SignalWindow:
    def __init__(self, s_on, s_off, side):
        self.s_on, self.s_off, self.side = s_on, s_off, side

    def __call__(self, s, t):
        return self.side if self.s_on <= s < self.s_off else 0


class ScriptedDriver:
    def __init__(self, board, speed, offset, signal):
        self.board, self.speed, self.offset, self.signal = board, speed, offset, signal
        self.pp = rs.control.PurePursuit()
        self.pi = rs.control.LongPI()

    def act(self, state, info, dt):
        route = self.board.route
        if info is None:
            s, lat, v, t, idx = 0.0, 0.0, 0.0, 0.0, 0
        else:
            s, lat, v, t, idx = info.s, info.lateral, info.v, info.t, info.index
        path = []
        for i in range(idx, min(len(route.pts), idx + AHEAD_PTS)):
            h = route.heading_at(i)
            off = self.offset(route.cum[i], t)
            x, y = route.pts[i]
            path.append((x - off * math.sin(h), y + off * math.cos(h)))
        state.speed = v
        return rs.Command(steer=self.pp.steer(state.x, state.y, state.heading, v, path),
                          accel=self.pi.accel(self.speed(s, t), v, dt),
                          turn=int(self.signal(s, t)), reason="SCRIPT", cap_by="SCRIPT",
                          lane_offset=self.offset(s, t), d_ego=lat)


class TeacherDriver:
    """ShadowTeacher 를 운전자 인터페이스로 감싼다. now 는 world.clock 이다."""

    def __init__(self, board):
        from vtd_rl.teacher.shadow import ShadowTeacher
        self.teacher = ShadowTeacher(board)
        self.world = None

    def bind(self, world):
        self.world = world

    def act(self, state, info, dt):
        return self.teacher.act(state, self.world.clock)
