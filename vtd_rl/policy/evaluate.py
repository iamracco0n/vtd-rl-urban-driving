"""학생 단독 주행 성적 — 완주율과 대회 점수(스펙 §6.4).

점수는 환경이 낸 구간 점수의 평균이다. 환경의 감점표는 M2b 에서 채점기와 같음을 확인했다.
"""
from dataclasses import dataclass

from vtd_rl.env.drive_env import EnvConfig, VtdDriveEnv
from vtd_rl.env.teacher_policy import run_teacher_in_env


@dataclass
class EpisodeOutcome:
    board: str
    seed: int
    outcome: str
    steps: int
    reward: float
    score: float
    sheet: list


def _summary(episodes) -> dict:
    n = max(len(episodes), 1)
    # 완주하지 못한 판은 0점으로 친다 — 멈춰 선 차는 위반을 안 해 점수가 오히려 높다(M3 관찰).
    scored = [e.score if e.outcome == "goal" else 0.0 for e in episodes]
    return {"goal_rate": sum(1 for e in episodes if e.outcome == "goal") / n,
            "mean_score": sum(scored) / n,
            "mean_score_raw": sum(e.score for e in episodes) / n,
            "mean_reward": sum(e.reward for e in episodes) / n,
            "episodes": episodes}


def _outcome_from_info(board, seed, info, steps, reward) -> EpisodeOutcome:
    result = info.get("result") or {"score": [0.0], "sheet": []}
    scores = result["score"] or [0.0]
    return EpisodeOutcome(board, seed, info["outcome"], steps, reward,
                          sum(scores) / len(scores), result["sheet"])


def run_policy_episode(env, policy, board_name: str, seed: int, max_steps: int = 20000):
    obs, info = env.reset(seed=seed, options={"board": board_name})
    total, steps = 0.0, 0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(policy.act(obs, deterministic=True))
        total += reward
        steps += 1
        if terminated or truncated:
            break
    return _outcome_from_info(board_name, seed, info, steps, total)


def evaluate_policy(policy, boards, seeds=(0, 1, 2), config: EnvConfig | None = None) -> dict:
    env = VtdDriveEnv(list(boards), config or EnvConfig())
    try:
        episodes = [run_policy_episode(env, policy, b.name, s) for b in boards for s in seeds]
    finally:
        env.close()
    return _summary(episodes)


def violation_counts(ev: dict) -> dict:
    """항목 -> {minor 수, major 수} — 구간-슬롯 단위로 센다(판마다 5구간, 항목은 구간별 채점표)."""
    counts: dict = {}
    for e in ev["episodes"]:
        for section in e.sheet:
            for item, grade in section.items():
                d = counts.setdefault(item, {"minor": 0, "major": 0})
                if grade in d:
                    d[grade] += 1
    return counts


def evaluate_teacher(boards, seeds=(0,), config: EnvConfig | None = None) -> dict:
    env = VtdDriveEnv(list(boards), config or EnvConfig())
    episodes = []
    try:
        for b in boards:
            for s in seeds:
                out = run_teacher_in_env(env, seed=s, options={"board": b.name})
                episodes.append(_outcome_from_info(b.name, s, out["info"], out["steps"],
                                                   out["reward"]))
    finally:
        env.close()
    return _summary(episodes)
