"""선생님으로 한 판 끝까지 달리기."""
import time
from dataclasses import dataclass

from vtd_rl import rule_stack as rs
from vtd_rl.teacher.shadow import ShadowTeacher
from vtd_rl.world.world import World


@dataclass
class EpisodeResult:
    board: str
    outcome: str
    sim_time: float
    steps: int
    wall_time: float
    distance: float

    @property
    def steps_per_sec(self) -> float:
        return self.steps / max(self.wall_time, 1e-9)


def run_teacher_episode(board, config=None, log_csv=None, seed=0) -> EpisodeResult:
    world = World(board, config, seed=seed)
    state = world.reset()
    teacher = ShadowTeacher(board)
    logger = rs.RunLogger(log_csv)
    steps, info = 0, None
    t0 = time.perf_counter()
    try:
        while info is None or not info.done:
            cmd = teacher.act(state, world.clock)
            logger.log(world.t, state, cmd)
            state, info = world.step(cmd.steer, cmd.accel, cmd.turn)
            steps += 1
    finally:
        logger.close()
    return EpisodeResult(board.name, info.outcome, info.t, steps, time.perf_counter() - t0, info.s)
