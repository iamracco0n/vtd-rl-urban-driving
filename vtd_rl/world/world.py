"""오프라인 세계 — 판 하나를 20 Hz 로 굴려 9910 과 같은 State 를 낸다."""
import random
from dataclasses import dataclass, field

from vtd_rl import rule_stack as rs
from vtd_rl.world import dynamics as dyn
from vtd_rl.world.actors import ActorSet, nearest_objects
from vtd_rl.world.signals import SignalReporter, programs_for, route_signals


@dataclass
class WorldConfig:
    dt: float = 0.05
    dynamics: dyn.DynamicsParams = field(default_factory=dyn.DynamicsParams)
    goal_radius: float = 10.0        # 경로 끝까지 남은 거리가 이 안이면 완주[m] — 완주 판정 반경 10~20 m 의 좁은 쪽
    offroad_margin: float = 1.5      # 그쪽 여유 폭(max(합법 l/r, 물리 pl/pr)) 밖으로 이만큼 나가면 도로 이탈[m]
    stall_seconds: float = 60.0      # 이만큼 진행이 없으면 정체[s]
    stall_progress: float = 1.0      # 이 이상 나아가야 진행으로 친다[m]
    time_limit_scale: float = 1.0    # 시나리오 duration 에 곱한다
    object_range: float = 80.0       # 9910 물체 수평 범위[m]
    max_objects: int = 30            # 9910 물체 슬롯 수
    clock_origin: float = 1.0e9      # 규칙 스택 now 오프셋 — 실차에서는 벽시계 epoch 가 들어간다


def _room(plan, legal, phys, default=1.75):
    """그쪽으로 갈 수 있는 폭[m] = max(합법, 물리) — 규칙 스택 `DrivingStack._phys_room` 과 같다. 둘 다 없으면 default."""
    return max(plan.get(legal) or 0.0, plan.get(phys) or 0.0) or default


@dataclass
class StepInfo:
    t: float
    s: float
    lateral: float
    index: int
    v: float
    done: bool
    outcome: str


class World:
    def __init__(self, board, config=None, signals=None, seed=0):
        self.board = board
        self.cfg = config or WorldConfig()
        if signals is None:
            db = rs.map_db()
            signals = route_signals(board.route, db["tl_map"], db["stoplines_all"])
        self.signals = signals
        self.seed = seed
        self.actors = ActorSet(board.scenario.actors)
        self.reset()

    @property
    def clock(self) -> float:
        return self.cfg.clock_origin + self.t

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        rng = random.Random(self.seed)
        x, y, h = self.board.start_pose
        self.ego = dyn.EgoState(x, y, h)
        self.t = 0.0
        self.turn = rs.TS_OFF
        self.actors.reset()
        self.reporter = SignalReporter(self.signals, self.board.lane_plan,
                                       programs_for(self.board.signals, self.signals, rng))
        self._hint = None
        p = self._project()
        self._best_s, self._best_t = p.s, 0.0
        self.time_limit = self.board.scenario.duration * self.cfg.time_limit_scale
        return self._observe(p, 0.0)

    def step(self, steer: float, accel: float, turn: int):
        self.ego = dyn.step(self.ego, steer, accel, self.cfg.dt, self.cfg.dynamics)
        self.t += self.cfg.dt
        self.turn = int(turn)
        p = self._project()
        state = self._observe(p, self.cfg.dt)
        outcome = self._outcome(p)
        return state, StepInfo(self.t, p.s, p.lateral, p.index, self.ego.v, outcome != "running", outcome)

    def _project(self):
        p = self.board.route.project(self.ego.x, self.ego.y, hint=self._hint)
        self._hint = p.index
        return p

    def _observe(self, p, dt):
        objs = self.actors.step(self.t, self.ego.x, self.ego.y, dt)
        objs = nearest_objects(objs, self.ego.x, self.ego.y, self.cfg.object_range, self.cfg.max_objects)
        tl_id, tl_state = self.reporter.report(p.s, p.index, self.t)
        return rs.State(x=self.ego.x, y=self.ego.y, heading=self.ego.heading, objects=objs,
                        speed=self.ego.v, speed_raw=self.ego.v,
                        tl_id=tl_id, tl_state=tl_state, t=self.clock)

    def _outcome(self, p) -> str:
        if self.board.route.total - p.s <= self.cfg.goal_radius:
            return "goal"
        plan = self.board.lane_plan[p.index] or {}
        if not (plan.get("j") or plan.get("jx")):
            left = _room(plan, "l", "pl") + self.cfg.offroad_margin
            right = _room(plan, "r", "pr") + self.cfg.offroad_margin
            if p.lateral > left or p.lateral < -right:
                return "offroad"
        if self.t >= self.time_limit - 1e-9:
            return "timeout"
        if p.s > self._best_s + self.cfg.stall_progress:
            self._best_s, self._best_t = p.s, self.t
        elif self.t - self._best_t >= self.cfg.stall_seconds - 1e-9:
            return "stalled"
        return "running"
