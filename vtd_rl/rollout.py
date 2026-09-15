"""운전자로 한 판 끝까지 달리기 — 행 기록 포함."""
import time
from dataclasses import dataclass

from vtd_rl.referee.rows import RowRecorder
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


@dataclass
class EpisodeRun:
    result: EpisodeResult
    rows: list


def run_episode(board, driver, config=None, log_csv=None, seed=0) -> EpisodeRun:
    world = World(board, config, seed=seed)
    state = world.reset()
    if hasattr(driver, "bind"):
        driver.bind(world)
    recorder = RowRecorder(log_csv)
    rows, info = [], None
    t0 = time.perf_counter()
    try:
        while info is None or not info.done:
            cmd = driver.act(state, info, world.cfg.dt)
            rows.append(recorder.record(world.t, state, cmd))
            state, info = world.step(cmd.steer, cmd.accel, cmd.turn)
    finally:
        recorder.close()
    result = EpisodeResult(board.name, info.outcome, info.t, len(rows), time.perf_counter() - t0, info.s)
    return EpisodeRun(result, rows)


def run_teacher_episode(board, config=None, log_csv=None, seed=0) -> EpisodeResult:
    from vtd_rl.drivers.scripted import TeacherDriver
    return run_episode(board, TeacherDriver(board), config, log_csv, seed).result
