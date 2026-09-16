"""Gymnasium 환경 — 세계(M1) + 온라인 심판(M2a) + 관측·행동·보상.

한 걸음(기본 10 Hz)은 시뮬 두 프레임이다. 심판과 행 기록은 **프레임마다** 돈다 —
채점기와 같은 판정을 내려면 20 Hz 행이 필요하고, 그림자 선생님도 프레임마다 같은 입력을 받아야 한다.
"""
import os
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from vtd_rl import rule_stack as rs
from vtd_rl.env.action import ActionConfig, action_space, frames_per_step, to_command
from vtd_rl.env.board_index import board_index
from vtd_rl.env.observation import ObsConfig, build_observation, observation_space
from vtd_rl.env.reward import COLLISION_ITEMS, RewardConfig, RewardShaper
from vtd_rl.referee.core import Referee
from vtd_rl.referee.rows import RowRecorder
from vtd_rl.world.world import World, WorldConfig

RUNNING = "running"


@dataclass
class EnvConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sections: int = 5
    use_map: bool = True
    log_dir: str | None = None


class VtdDriveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, boards, config: EnvConfig | None = None, render_mode=None):
        if not boards:
            raise ValueError("판이 하나는 있어야 한다")
        self.boards = list(boards)
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.observation_space = observation_space(self.cfg.obs)
        self.action_space = action_space(self.cfg.action)
        self.frames = frames_per_step(self.cfg.action, self.cfg.world.dt)
        self.frame_hook = None
        self.board = self.world = self.referee = self.state = None
        self._worlds: dict = {}          # 판마다 세계를 다시 짓지 않는다(신호 찾기가 판당 수십 ms)
        self._episode = 0
        self._recorder = None
        self._info = None
        self._prev_action = self._zero_action()
        self._shaper = None

    # ------------------------------------------------------------------ Gymnasium
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_recorder()
        name = (options or {}).get("board")
        if name is not None:
            self.board = next(b for b in self.boards if b.name == name)
        else:
            self.board = self.boards[int(self.np_random.integers(len(self.boards)))]
        world_seed = int(self.np_random.integers(2 ** 31 - 1))
        self.world = self._worlds.get(self.board.name)
        if self.world is None:
            idx = board_index(self.board)
            self.world = self._worlds[self.board.name] = World(
                self.board, self.cfg.world, signals=idx.signals, seed=world_seed)
        self.state = self.world.reset(world_seed)
        self.referee = Referee(self.board, self.cfg.sections, self.cfg.use_map)
        self._shaper = RewardShaper(self.board, self.cfg.reward)
        self._shaper.reset()
        self._info = None
        self._prev_action = self._zero_action()
        self._episode += 1
        self._recorder = RowRecorder(self._csv_path())
        obs = build_observation(self.world, self.state, self._info, self._prev_pair(), self.cfg.obs)
        return obs, {"board": self.board.name, "outcome": RUNNING, "s": 0.0, "sim_time": 0.0,
                     "frames": 0, "hits": [], "counted": 0, "reward_terms": {}}

    def step(self, action):
        steer, accel, turn = to_command(action, self.cfg.action)
        cmd = rs.Command(steer=steer, accel=accel, turn=turn, reason="RL", cap_by="RL")
        s0 = self._info.s if self._info is not None else 0.0
        hits, frames, outcome = [], 0, RUNNING
        for _ in range(self.frames):
            if self.frame_hook is not None:
                self.frame_hook(self.state, self.world.clock)
            cmd.d_ego = self._info.lateral if self._info is not None else 0.0
            self.state.speed = self.world.ego.v          # 로그 속도의 주인은 세계다
            hits += self.referee.step(self._recorder.record(self.world.t, self.state, cmd))
            self.state, self._info = self.world.step(steer, accel, turn)
            frames += 1
            if self._info.done:
                outcome = self._info.outcome
                break
        if outcome != RUNNING or any(h.item in COLLISION_ITEMS for h in hits):
            hits += self.referee.finish()          # 남은 대기 판정까지 이 걸음의 보상에 넣는다
        shaped = self._shaper.step(hits, max(0.0, self._info.s - s0), action, self._prev_action,
                                   outcome)
        self._prev_action = {"control": np.asarray(action["control"], dtype=np.float32).copy(),
                             "turn": int(action["turn"])}
        terminated = outcome in ("goal", "offroad") or shaped.collision
        truncated = outcome in ("timeout", "stalled")
        info = {"board": self.board.name, "outcome": outcome, "s": self._info.s,
                "sim_time": self._info.t, "frames": frames,
                "hits": [(h.t, h.sec, h.item, h.level) for h in hits],
                "counted": shaped.counted, "reward_terms": shaped.terms}
        if terminated or truncated:
            info["episode"] = self._finish()
        obs = build_observation(self.world, self.state, self._info, self._prev_pair(), self.cfg.obs)
        return obs, float(shaped.total), bool(terminated), bool(truncated), info

    def close(self):
        self._close_recorder()

    # ------------------------------------------------------------------ 안쪽
    def _zero_action(self):
        return {"control": np.zeros(2, dtype=np.float32), "turn": 0}

    def _prev_pair(self):
        return (float(self._prev_action["control"][0]), float(self._prev_action["control"][1]),
                int(self._prev_action["turn"]))

    def _csv_path(self):
        if not self.cfg.log_dir:
            return None
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        return os.path.join(self.cfg.log_dir, f"{self.board.name}-{self._episode:04d}.csv")

    def _close_recorder(self):
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def _finish(self):
        self._close_recorder()
        sheet = self.referee.sheet
        return {"score": [sheet.score(k) for k in range(sheet.n)],
                "sheet": [dict(s) for s in sheet.state],
                "respawns": dict(self.referee.respawns),
                "notes": list(self.referee.ctx.notes),
                "rows_csv": self._csv_path()}
