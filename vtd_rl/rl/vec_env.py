"""벡터 환경 — 판을 여러 프로세스에서 동시에 굴린다.

판 목록은 단계 ①(항상 초록)과 ②(신호 주기)를 **섞는다**. M3 는 단계 ①만으로 배워 단계 ②가
제로샷이었고, 그래서 학생이 빨간불을 무시했다(M3 성적표). PPO 는 두 단계를 함께 본다.
"""
import os

import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.policy.encode import VEC_KEYS
from vtd_rl.world.board import load_curriculum

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _boards(curricula):
    """커리큘럼 파일들을 읽어 한 환경이 뽑을 판 목록을 만든다.

    같은 코스가 단계마다 같은 이름으로 다시 나온다 — `curricula/stage1.json` 과
    `stage2.json` 은 둘 다 `course_A`~`course_H` 를 쓰고 신호 방식(`signals`)만 다르다.
    `VtdDriveEnv` 의 세계 캐시(`_worlds`)는 판 **이름**만을 열쇠로 쓰므로, 이름이 같은
    두 판을 그대로 섞으면 한 판을 골랐다가 다른 판을 고르는 순간
    `AssertionError: 세계 캐시가 다른 판을 준다`로 죽는다(2026-09-21, 이 커리큘럼 조합으로
    실측 재현). 판 이름에 커리큘럼 파일 이름을 붙여 물리 코스는 그대로 두고 신호 단계만
    갈라놓는다 — 심판·세계·환경의 판정에는 손대지 않는, 이 안에서 끝나는 수선이다.
    """
    out = []
    for rel in curricula:
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        stage = os.path.splitext(os.path.basename(path))[0]
        _name, boards = load_curriculum(path)
        for board in boards:
            board.name = f"{board.name}#{stage}"
        out += boards
    return out


def make_env_fn(curricula, config: EnvConfig | None, seed: int, rank: int):
    def _make():
        # 시드는 make_vec_env 의 reset(seed=[...]) 이 준다 — 여기서 리셋하면 세계를 한 번 더 짓는다.
        return VtdDriveEnv(_boards(curricula), config or EnvConfig())
    return _make


def make_vec_env(curricula, n_envs: int, config: EnvConfig | None = None, seed: int = 0,
                 asynchronous: bool = True):
    fns = [make_env_fn(tuple(curricula), config, seed, i) for i in range(n_envs)]
    venv = AsyncVectorEnv(fns) if asynchronous else SyncVectorEnv(fns)
    venv.reset(seed=[seed + i for i in range(n_envs)])
    return venv


def vec_obs_to_arrays(obs: dict):
    n = len(obs["ego"])
    vec = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(n, -1)
                          for k in VEC_KEYS], axis=1)
    return (vec, np.asarray(obs["objects"], dtype=np.float32),
            np.asarray(obs["object_mask"], dtype=np.float32))
