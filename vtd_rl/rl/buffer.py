"""롤아웃 저장과 GAE.

시간 초과·정체(truncated)는 판이 끝난 게 아니므로 가치로 부트스트랩하고, 종료(terminated)만
끊는다 — 환경이 둘을 나눠 주므로(그 구분은 M2b 에서 배타적으로 만들어 두었다) 그대로 쓴다.
`.add()` 의 `done` 인자에는 **반드시 terminated 만** 넘겨야 한다(truncated 를 섞으면 정체로
끝난 판까지 부트스트랩이 끊겨 가치를 과소평가한다) — 그 구분은 학습 루프(Task 6)의 책임이다.

gymnasium 1.3 의 벡터 환경은 `NEXT_STEP` 자동 리셋이라, 판이 끝난 **다음 걸음**에서 행동이
무시되고 리셋만 일어나며 보상 0·terminated=False 가 돌아온다. 그 한 걸음은 진짜 전이가
아니므로 `valid=0` 으로 표시해 받고(GAE 재귀는 앞 걸음의 done=1 이 이미 끊어 주니 손대지
않는다), 미니배치를 만들 때 빼 버린다.
"""
import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, device):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        self.reset()

    def reset(self):
        self._rows = []
        self.advantages_raw = None
        self.returns = None

    def add(self, **row):
        self._rows.append({k: v.detach().to(self.device) for k, v in row.items()})

    def _stack(self, key):
        return torch.stack([r[key] for r in self._rows])            # [T, N, ...]

    def compute_gae(self, last_value, gamma: float = 0.99, lam: float = 0.95):
        rewards, values, dones = self._stack("reward"), self._stack("value"), self._stack("done")
        adv = torch.zeros_like(rewards)
        running = torch.zeros(self.n_envs, device=self.device)
        next_value = last_value.to(self.device)
        for t in reversed(range(len(self._rows))):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            running = delta + gamma * lam * not_done * running
            adv[t] = running
            next_value = values[t]
        self.advantages_raw = adv                     # [T, N]
        self.returns = adv + values                   # [T, N]
        return self.advantages_raw

    def batches(self, minibatch: int, generator=None):
        flat = {k: self._stack(k).flatten(0, 1) for k in self._rows[0]}
        keep = flat["valid"].nonzero(as_tuple=True)[0]        # 자동 리셋 더미를 뺀다
        flat = {k: v[keep] for k, v in flat.items()}
        adv = self.advantages_raw.flatten(0, 1)[keep]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = self.returns.flatten(0, 1)[keep]
        order = torch.randperm(adv.shape[0], generator=generator)
        for start in range(0, adv.shape[0], minibatch):
            idx = order[start:start + minibatch].to(self.device)
            yield (flat["vec"][idx], flat["objs"][idx], flat["mask"][idx], flat["raw"][idx],
                   flat["turn"][idx], flat["log_prob"][idx], adv[idx], ret[idx],
                   flat["value"][idx])
