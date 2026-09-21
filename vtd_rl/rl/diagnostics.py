"""학습이 무엇을 하고 있는지 보이게 하는 계측.

M4a 는 세 실행이 1M~1.5M 이후 무너졌는데 **왜** 무너졌는지 못 밝혔다. 이유는 계측이 없어서다:
PPO 가 실제로 최대화하는 **확률적 롤아웃 리턴**이 한 번도 로깅되지 않아, 인용할 수 있는
보상은 결정적 평가 보상(프록시)뿐이었다. 종료 사유와 정책 드리프트도 마찬가지로 없었다.

세 가지를 잰다 — 롤아웃 리턴(PPO 의 목적함수), 종료 사유(붕괴가 충돌·이탈·정체 중 무엇으로
나타나는가), 초기 정책으로부터의 거리(정책이 얼마나 멀리 갔는가).
"""
import numpy as np
import torch

# 판이 아직 안 끝난 상태 — 종료 사유가 아니므로 세지 않는다.
RUNNING = "running"


class ReturnTracker:
    """환경별로 보상을 누적하다가 판이 끝나면 그 리턴을 담는다.

    `reset()` 은 **담아 둔 리턴만** 비운다. 진행 중인 누적은 롤아웃 경계를 넘어 이어져야 한다 —
    판 하나가 롤아웃 여러 개에 걸치기 때문이다(판은 2500~5000 걸음, 롤아웃은 256 걸음).
    """

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self._running = np.zeros(n_envs, dtype=np.float64)
        self._steps = np.zeros(n_envs, dtype=np.int64)
        self.reset()

    def reset(self):
        self._returns: list = []
        self._lengths: list = []

    def add(self, reward, done):
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        self._running += reward
        self._steps += 1
        for i in np.nonzero(done)[0]:
            self._returns.append(float(self._running[i]))
            self._lengths.append(int(self._steps[i]))
            self._running[i] = 0.0
            self._steps[i] = 0

    def stats(self) -> dict:
        n = len(self._returns)
        return {"rollout_return_mean": (sum(self._returns) / n) if n else None,
                "rollout_return_n": n,
                "rollout_len_mean": (sum(self._lengths) / n) if n else None}


class OutcomeCounter:
    """벡터 환경 `infos` 에서 종료 사유를 센다 — 붕괴가 무엇으로 나타나는지 보려는 것."""

    def __init__(self):
        self._counts: dict = {}

    def add(self, infos: dict):
        values = infos.get("outcome")
        if values is None:
            return
        for v in np.asarray(values, dtype=object).reshape(-1):
            if v is None or v == RUNNING:
                continue
            key = f"outcome_{v}"
            self._counts[key] = self._counts.get(key, 0) + 1

    def stats(self) -> dict:
        return dict(self._counts)


def snapshot_policy(net) -> dict:
    """정책 파라미터를 떼어 복사한다 — clone 을 빼면 이후 갱신이 기준까지 따라 움직인다."""
    return {k: v.detach().clone() for k, v in net.policy.state_dict().items()}


def policy_drift(net, ref_state: dict) -> dict:
    """현재 정책이 기준에서 얼마나 멀어졌나. `approx_kl` 은 갱신 한 걸음의 크기만 보지만,

    이 값은 **누적된 이동**을 본다 — M4a 의 붕괴는 걸음마다는 작고(KL 0.011~0.016) 누적으로만
    큰 형태였다.
    """
    with torch.no_grad():
        sq, ref_sq = 0.0, 0.0
        for k, v in net.policy.state_dict().items():
            r = ref_state[k].to(v.device)
            sq += float((v - r).pow(2).sum())
            ref_sq += float(r.pow(2).sum())
    l2 = sq ** 0.5
    return {"drift_l2": l2, "drift_rel": l2 / (ref_sq ** 0.5) if ref_sq > 0 else 0.0}
