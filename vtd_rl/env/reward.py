"""보상 — 스펙 §5. 심판이 낸 감점과 진행 거리로 매 걸음 보상을 만든다.

대회는 같은 항목을 구간당 한 번만 깎지만 학습 보상은 위반마다 깎는다. 다만 심판은 채점기와 같은
호출을 다 내므로(과속 한 판에 198번) 같은 항목이 이어지면 repeat_gap 초에 한 번만 센다 —
'한 번 깎였으니 계속 어겨도 된다' 와 '프레임마다 깎여서 다른 항이 묻힌다' 사이를 가른다.
"""
from dataclasses import dataclass, field

COLLISION_ITEMS = (11, 14)       # score_fma ⑪ 장애물 충돌 · ⑭ 차량·보행자 접촉


@dataclass(frozen=True)
class RewardConfig:
    progress_total: float = 100.0    # 완주까지 진행 항의 합
    goal_bonus: float = 50.0
    time_cost: float = 0.01          # 판단 한 걸음마다
    minor: float = -3.0
    major: float = -6.0
    rule_scale: float = 1.0          # 커리큘럼 단계마다 키운다(스펙 §5)
    collision: float = -50.0
    offroad: float = -50.0
    comfort_steer: float = -0.10
    comfort_accel: float = -0.05
    repeat_gap: float = 1.0          # 같은 항목을 다시 세기까지[s]


@dataclass
class RewardStep:
    total: float
    terms: dict
    collision: bool
    counted: int


class ViolationTracker:
    def __init__(self, repeat_gap: float):
        self.repeat_gap = repeat_gap
        self._last: dict = {}

    def reset(self):
        self._last.clear()

    def count(self, hits):
        out = []
        for h in hits:
            last = self._last.get(h.item)
            if last is None or h.t - last >= self.repeat_gap - 1e-9:
                self._last[h.item] = h.t
                out.append(h)
        return out


@dataclass
class RewardShaper:
    board: object
    cfg: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        self.tracker = ViolationTracker(self.cfg.repeat_gap)
        self._per_m = self.cfg.progress_total / max(self.board.route.total, 1.0)

    def reset(self):
        self.tracker.reset()

    def step(self, hits, ds: float, action, prev_action, outcome: str) -> RewardStep:
        cfg = self.cfg
        counted = self.tracker.count(hits)
        collision = any(h.item in COLLISION_ITEMS for h in counted)
        violation = sum(cfg.major if h.level == "major" else cfg.minor
                        for h in counted if h.item not in COLLISION_ITEMS) * cfg.rule_scale
        d_steer = abs(float(action["control"][0]) - float(prev_action["control"][0]))
        d_accel = abs(float(action["control"][1]) - float(prev_action["control"][1]))
        terms = {
            "progress": self._per_m * ds,
            "time": -cfg.time_cost,
            "violation": violation,
            "collision": cfg.collision if collision else 0.0,
            "offroad": cfg.offroad if outcome == "offroad" else 0.0,
            "goal": cfg.goal_bonus if outcome == "goal" else 0.0,
            "comfort": cfg.comfort_steer * d_steer + cfg.comfort_accel * d_accel,
        }
        return RewardStep(sum(terms.values()), terms, collision, len(counted))
