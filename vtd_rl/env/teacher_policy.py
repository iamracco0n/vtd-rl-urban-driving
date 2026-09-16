"""그림자 선생님 — 규칙 스택을 환경 행동으로(스펙 §6.1).

규칙 스택은 추월 상태기계·타이머 같은 내부 상태가 있어서, 학생이 운전하는 판에서도 **매 프레임**
같은 입력을 먹여야 계산이 이어진다. 그래서 env.frame_hook 에 걸어 프레임마다 돌리고,
판단 시점에는 가장 최근 명령을 행동으로 바꿔 준다.

명령의 `reason`/`cap_by` 도 그대로 env 에 넘긴다(env.command_tags). 채점기 ⑧ 은 이 두 칸으로
안전 정차를 면책하므로, "RL" 로 덮으면 선생님이 앞차 뒤에 선 것까지 감점으로 찍힌다.
"""
from vtd_rl.env.action import from_command
from vtd_rl.env.tags import RL
from vtd_rl.teacher.shadow import ShadowTeacher


class TeacherPolicy:
    def __init__(self, env):
        self.env = env
        self.teacher = None
        self.board_name = None
        self.episode = None
        self.command = None
        env.frame_hook = self._on_frame
        env.command_tags = self._tags

    def detach(self):
        """환경에서 갈고리를 뗀다 — 선생님이 물러난 판에까지 따라가지 않게.

        떼지 않으면 학생이 운전하는 판에서도 규칙 스택이 **매 프레임** 돌고(그만큼 느리다),
        무엇보다 행의 `reason`/`cap_by` 에 **운전하지도 않은 선생님의 사유**가 찍힌다 —
        심판 ⑧ 면책이 엉뚱한 근거로 붙거나 빠진다.
        """
        if self.env.frame_hook == self._on_frame:      # 묶인 메서드는 볼 때마다 새 객체다(is 로는 못 본다)
            self.env.frame_hook = None
        if self.env.command_tags == self._tags:
            self.env.command_tags = None

    def reset(self):
        """규칙 스택을 새로 짓는다 — 판이 바뀌었을 때도, 같은 판을 다시 시작했을 때도.

        `ShadowTeacher` 는 추월 상태기계와 `_last_now` 를 들고 있다. 판 이름만 보고 재사용하면
        새 판이 지난 판의 상태기계를 이어받고 `dt` 가 1e-3 바닥으로 무너져 **같은 판인데 결과가
        달라진다**(실측: 219 -> 221 걸음, 보상 143.6554 -> 143.5132).
        """
        self.teacher = ShadowTeacher(self.env.board)
        self.board_name = self.env.board.name
        self.episode = self.env.episode
        self.command = None

    def _stale(self):
        return (self.teacher is None or self.board_name != self.env.board.name
                or self.episode != self.env.episode)

    def _on_frame(self, state, clock):
        if self._stale():
            self.reset()
        self.command = self.teacher.act(state, clock)

    def _tags(self):
        if self.command is None:
            return RL, RL
        return self.command.reason, self.command.cap_by

    def act(self):
        if self.command is None or self._stale():      # 첫 걸음 — 아직 프레임 훅이 안 돌았다
            self._on_frame(self.env.state, self.env.world.clock)
        cmd = self.command
        return from_command(cmd.steer, cmd.accel, cmd.turn, self.env.cfg.action)


def run_teacher_in_env(env, seed=None, options=None, max_steps=20000) -> dict:
    _obs, info = env.reset(seed=seed, options=options)
    policy = TeacherPolicy(env)
    policy.reset()
    total, terms, steps = 0.0, {}, 0
    try:
        for _ in range(max_steps):
            _obs, reward, terminated, truncated, info = env.step(policy.act())
            total += reward
            steps += 1
            for k, v in info["reward_terms"].items():
                terms[k] = terms.get(k, 0.0) + v
            if terminated or truncated:
                break
    finally:
        policy.detach()      # 이 판만 선생님이 몬다 — 다음 판까지 갈고리를 끌고 가지 않는다
    return {"outcome": info["outcome"], "steps": steps, "sim_time": info["sim_time"],
            "reward": total, "terms": terms, "info": info}
