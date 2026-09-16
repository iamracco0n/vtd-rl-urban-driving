"""그림자 선생님 — 규칙 스택을 환경 행동으로(스펙 §6.1).

규칙 스택은 추월 상태기계·타이머 같은 내부 상태가 있어서, 학생이 운전하는 판에서도 **매 프레임**
같은 입력을 먹여야 계산이 이어진다. 그래서 env.frame_hook 에 걸어 프레임마다 돌리고,
판단 시점에는 가장 최근 명령을 행동으로 바꿔 준다.
"""
from vtd_rl.env.action import from_command
from vtd_rl.teacher.shadow import ShadowTeacher


class TeacherPolicy:
    def __init__(self, env):
        self.env = env
        self.teacher = None
        self.board_name = None
        self.command = None
        env.frame_hook = self._on_frame

    def reset(self):
        self.teacher = ShadowTeacher(self.env.board)
        self.board_name = self.env.board.name
        self.command = None

    def _on_frame(self, state, clock):
        if self.teacher is None or self.board_name != self.env.board.name:
            self.reset()
        self.command = self.teacher.act(state, clock)

    def act(self):
        if self.command is None:                       # 첫 걸음 — 아직 프레임 훅이 안 돌았다
            self._on_frame(self.env.state, self.env.world.clock)
        cmd = self.command
        return from_command(cmd.steer, cmd.accel, cmd.turn, self.env.cfg.action)


def run_teacher_in_env(env, seed=None, options=None, max_steps=20000) -> dict:
    _obs, info = env.reset(seed=seed, options=options)
    policy = TeacherPolicy(env)
    policy.reset()
    total, terms, steps = 0.0, {}, 0
    for _ in range(max_steps):
        _obs, reward, terminated, truncated, info = env.step(policy.act())
        total += reward
        steps += 1
        for k, v in info["reward_terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        if terminated or truncated:
            break
    return {"outcome": info["outcome"], "steps": steps, "sim_time": info["sim_time"],
            "reward": total, "terms": terms, "info": info}
