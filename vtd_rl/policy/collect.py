"""한 판을 몰며 **선생님 정답**을 모은다(스펙 §6.2).

DAgger 의 핵심은 '학생이 간 상태에서의 선생님 답'이다. 그래서 실행 행동은 β 로 섞되,
라벨은 언제나 그 프레임의 선생님 행동이다. 선생님은 env.frame_hook 으로 매 시뮬 프레임 돌아
내부 상태를 이어 간다(스펙 §6.1) — 학생이 몰아도 마찬가지다.
"""
import numpy as np

from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.teacher_policy import TeacherPolicy
from vtd_rl.policy.dataset import Shard
from vtd_rl.policy.encode import flatten_obs


def collect_episode(board, policy=None, beta: float = 1.0, seed: int = 0,
                    config: EnvConfig | None = None, max_steps: int = 20000,
                    rng=None) -> Shard:
    env = VtdDriveEnv([board], config or EnvConfig())
    rng = rng or np.random.default_rng(seed)
    teacher = TeacherPolicy(env)
    vecs, objs, masks, controls, turns = [], [], [], [], []
    student_steps, total_reward = 0, 0.0
    try:
        obs, info = env.reset(seed=seed, options={"board": board.name})
        teacher.reset()
        for _ in range(max_steps):
            label = teacher.act()
            vec, obj, mask = flatten_obs(obs)
            vecs.append(vec)
            objs.append(obj)
            masks.append(mask)
            controls.append(np.asarray(label["control"], dtype=np.float32))
            turns.append(int(label["turn"]))
            if policy is not None and rng.random() >= beta:
                action = policy.act(obs, deterministic=True)
                student_steps += 1
            else:
                action = label
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    finally:
        teacher.detach()
        env.close()
    meta = {"board": board.name, "seed": int(seed), "beta": float(beta),
            "outcome": info["outcome"], "steps": len(turns),
            "reward": float(total_reward), "student_steps": int(student_steps)}
    return Shard(np.stack(vecs), np.stack(objs), np.stack(masks),
                 np.stack(controls), np.asarray(turns, dtype=np.int64), meta)
