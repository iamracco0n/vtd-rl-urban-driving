"""한 판을 몰며 **선생님 정답**을 모은다(스펙 §6.2).

DAgger 의 핵심은 '학생이 간 상태에서의 선생님 답'이다. 그래서 실행 행동은 β 로 섞되,
라벨은 언제나 그 프레임의 선생님 행동이다. 선생님은 env.frame_hook 으로 매 시뮬 프레임 돌아
내부 상태를 이어 간다(스펙 §6.1) — 학생이 몰아도 마찬가지다.

라벨은 **기록하는 관측과 같은 상태**에서 나온 선생님 명령이어야 한다(수정 라운드 1). 한 판단
스텝(`env.step()`)은 여러 시뮬 프레임을 밟는데, `teacher.act()` 가 돌려주는 `self.command` 는
그 스텝의 **마지막** frame_hook 호출 결과 — 즉 관측을 만든 마지막 `world.step()` 보다 한 프레임
전의 상태에서 나온 값이다. 그래서 선생님이 건 진짜 갈고리를 얇게 감싸, **이번 `env.step()` 의
첫 frame_hook 호출**(아직 세계가 한 프레임도 더 안 나간 시점 — 정확히 지금 관측의 상태)에서
나온 명령만 따로 받아 라벨로 쓴다. 몰기(행동 선택)는 그대로 `teacher.act()`(선생님이 몰 때) —
`run_teacher_in_env` 와 같은 값이라 완주 여부가 달라지지 않는다.
"""
import numpy as np

from vtd_rl.env.action import from_command
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
    teacher_hook = env.frame_hook          # 선생님이 건 진짜 갈고리(TeacherPolicy.__init__ 이 배선)
    holder = {"cmd": None}

    def _label_hook(state, clock):
        teacher_hook(state, clock)         # 선생님 계산은 그대로 — 매 프레임, 학생이 몰아도
        if holder["cmd"] is None:          # 이번 env.step() 의 첫 프레임만 잡는다 — 지금 관측의 상태다
            holder["cmd"] = teacher.command

    env.frame_hook = _label_hook

    vecs, objs, masks, controls, turns = [], [], [], [], []
    student_steps, total_reward, fallbacks = 0, 0.0, 0
    try:
        obs, info = env.reset(seed=seed, options={"board": board.name})
        teacher.reset()
        for _ in range(max_steps):
            vec, obj, mask = flatten_obs(obs)
            if policy is not None and rng.random() >= beta:
                action = policy.act(obs, deterministic=True)
                student_steps += 1
            else:
                action = teacher.act()     # 선생님이 몬다 — run_teacher_in_env 와 같은 행동 선택
            holder["cmd"] = None           # 이번 env.step() 의 첫 프레임을 기다린다
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if holder["cmd"] is not None:
                cmd = holder["cmd"]
                label = from_command(cmd.steer, cmd.accel, cmd.turn, env.cfg.action)
            else:
                label = action              # 프레임이 한 번도 안 돎(있어선 안 되는 경우) — 폴백
                fallbacks += 1
            vecs.append(vec)
            objs.append(obj)
            masks.append(mask)
            controls.append(np.asarray(label["control"], dtype=np.float32))
            turns.append(int(label["turn"]))
            if terminated or truncated:
                break
    finally:
        teacher.detach()
        env.frame_hook = None    # detach() 는 self._on_frame 과만 같음을 비교한다 — 여기서 감쌌으니 직접 뗀다
        env.close()
    if fallbacks:
        print(f"[collect] {board.name} seed={seed}: 라벨 폴백 {fallbacks}회(프레임 훅이 한 번도 안 돎)",
              flush=True)
    meta = {"board": board.name, "seed": int(seed), "beta": float(beta),
            "outcome": info["outcome"], "steps": len(turns),
            "reward": float(total_reward), "student_steps": int(student_steps)}
    return Shard(np.stack(vecs), np.stack(objs), np.stack(masks),
                 np.stack(controls), np.asarray(turns, dtype=np.int64), meta)
