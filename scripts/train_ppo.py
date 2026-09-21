"""M4a — PPO 학습 스크립트(가치 워밍업 → PPO, 평가·체크포인트·로그).

    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
        --out runs/omen/$(date +%F)-ppo --init runs/lab-main/.../policy-r4.pt \
        --dagger-data runs/lab-main/.../data --envs 30 --steps 3000000
    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py --smoke --out /tmp/smoke

체크포인트는 `ActorCritic`(가치 머리 포함)이지 `DrivePolicy` 가 아니다 — 이름을 `ac-<스텝>.pt`
로 두는 이유가 그것이다. `DrivePolicy.load()` 로는 못 읽는다, `ActorCritic.load(path)` 를 쓰고
정책만 필요하면 `.policy` 를 꺼낸다.

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json` 과 `stage2.json` 은 판 이름을
그대로 공유하고 `signals` 만 다르므로, `evaluate_policy` 에 두 단계 판을 합쳐 넘기면
`VtdDriveEnv` 의 세계 캐시가 "AssertionError: 세계 캐시가 다른 판을 준다" 로 죽는다
(2026-09-21, `vtd_rl/rl/vec_env.py` 의 실측 기록). 평가에 넘길 판은 `load_curriculum` 으로
직접 읽는다 — `vec_env._boards()` 는 이름에 `#stage1` 접미사를 붙이므로 쓰지 않는다.

학습 중 평가(주기 평가, `--eval-seeds`)는 값싸게(기본 1시드) 자주 하고, 학습이 끝난 뒤
`--final-eval-seeds`(기본 3)로 딱 한 번 전체 평가를 해서 그 값으로 `ac-best.pt` 를 최종
선정하고 요약 JSON 에 싣는다 — 실측(OMEN) 전체 평가 한 번 ≈ 81 초인데 3M 스텝의 환경 시간은
2.7 분뿐이라, 주기 평가를 매번 전체 시드로 하면 평가가 학습보다 몇 배 더 걸린다.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl.env.drive_env import EnvConfig  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.dataset import load_dir  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.train import TrainConfig  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.rl.buffer import RolloutBuffer  # noqa: E402
from vtd_rl.rl.ppo import PPOConfig, dagger_batches, update  # noqa: E402
from vtd_rl.rl.vec_env import make_vec_env, vec_obs_to_arrays  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

PUBLIC_KEYS = ("goal_rate", "mean_score", "mean_score_raw", "mean_reward")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def _stage_boards(curricula, smoke: bool):
    """커리큘럼 파일마다 (이름표, 판 목록) — 판 이름은 그대로 둔다(`vec_env._boards()` 와 달리

    `#stage` 접미사를 붙이지 않는다). 이름표는 파일 이름(확장자 없이)이라 기본 커리큘럼
    `stage1.json`/`stage2.json` 이면 요약 JSON 의 키가 그대로 "stage1"/"stage2" 가 된다.
    스모크에서는 각 단계 코스 A 한 판으로 줄인다.
    """
    stages = []
    for rel in curricula:
        path = _resolve(rel)
        label = os.path.splitext(os.path.basename(path))[0]
        _name, boards = load_curriculum(path)
        if smoke:
            boards = [b for b in boards if b.name == "course_A"]
        stages.append((label, boards))
    return stages


def _public(ev: dict) -> dict:
    return {k: ev[k] for k in PUBLIC_KEYS}


def _evaluate_stages(policy, stages, seeds) -> dict:
    return {label: evaluate_policy(policy, boards, seeds=seeds) for label, boards in stages}


def _metric(evs: dict) -> tuple:
    """단계 평균 완주율 → 동률이면 단계 평균 점수. `ac-best.pt` 선정·주기 최고 추적에 같이 쓴다."""
    n = max(len(evs), 1)
    goal = sum(e["goal_rate"] for e in evs.values()) / n
    score = sum(e["mean_score"] for e in evs.values()) / n
    return (goal, score)


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """비평가가 실제 리턴을 얼마나 따라잡는지 — 1 이면 완벽, 0 이면 평균만 맞히는 것과 같다.

    자동 리셋 더미 행(`valid=0`)은 호출부가 미리 걸러 넘긴다 — 그 행은 학습 배치에도 안
    들어가니 진단에도 안 들어가야 한다. 리턴 분산이 0 에 가까우면(모든 보상이 같은 병적인
    경우) 정의가 안 되니 NaN 을 낸다(호출부가 JSON 에 쓰기 전에 `None` 으로 바꾼다).
    """
    var_y = float(returns.var())
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - float((returns - values).var()) / var_y


def _bootstrap_reward_done(reward, term, prev_done, value: torch.Tensor):
    """자동 리셋 더미 행(직전 걸음이 끝난 환경)의 `reward`·`done` 을 고쳐 GAE 사슬을 끊는다.

    gymnasium 1.3 자동 리셋(NEXT_STEP)은 판이 끝난 다음 걸음에서 행동을 무시하고 보상 0·
    `terminated=False` 를 낸다. `RolloutBuffer.compute_gae` 는 `not_done` 하나로 부트스트랩
    허용과 사슬 절단을 동시에 정한다 — 이 더미 행의 `done` 을 그대로(=`term`, 거짓) 두면
    `delta = γ·V(새 판 첫 관측) − V(끝난 판 마지막 관측)` 이 계산되고, 그 값이 `(γλ)^k` 로
    감쇠하며 이전(진짜) 걸음들의 이점에 새어 든다 — truncated(시간 초과·정체)로 끝난 판일수록
    V(새 판 시작)이 크고 V(정체 판 마지막)은 작아 체계적으로 양의 이점이 새어나간다("판을
    빨리 포기하면 이득"을 학습하게 된다, 2026-09-21 리뷰 재현).

    더미 행의 `reward` 를 제 `value` 로, `done` 을 1 로 주면 `delta = value − value = 0` 이고
    `not_done = 0` 이라 사슬이 정확히 그 자리에서 끊긴다 — 이 행 자체는 `valid=0` 이라 학습
    배치에는 어차피 안 들어간다. 진짜 종료(terminated)·계속되는 걸음(`prev_done` 이 거짓인
    행)은 손대지 않는다.
    """
    dummy = torch.as_tensor(prev_done, device=value.device)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=value.device)
    done_t = torch.as_tensor(term, dtype=torch.float32, device=value.device)
    reward_t = torch.where(dummy, value.detach(), reward_t)
    done_t = torch.where(dummy, torch.ones_like(done_t), done_t)
    return reward_t, done_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None,
                    help="M3 DrivePolicy 체크포인트로 정책을 워밍스타트한다(가치 머리는 새로 시작)")
    ap.add_argument("--curricula", nargs="+",
                    default=[os.path.join(REPO, "curricula", "stage1.json"),
                            os.path.join(REPO, "curricula", "stage2.json")])
    ap.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="lab-main 14, OMEN 30 — 실측 기준(cpu_count - 2)")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--warmup-updates", type=int, default=5,
                    help="처음 이 갱신 수만큼은 정책 기울기를 끄고 가치·모방만 학습한다(스펙 §6.3)")
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--eval-seeds", type=int, default=1,
                    help="학습 중 주기 평가 시드 수 — 값싸게, 자주(로그·체크포인트 판단용)")
    ap.add_argument("--final-eval-seeds", type=int, default=3,
                    help="학습이 끝난 뒤 딱 한 번의 전체 평가 시드 수 — 요약 JSON·ac-best.pt 선정에 쓴다")
    ap.add_argument("--dagger-data", default=None, help="M3 가 모은 조각 폴더(runs/.../data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="환경 2개·스텝 4000·롤아웃 64·동기 벡터 환경·평가는 각 단계 코스 A 한 판씩 시드 1개")
    a = ap.parse_args()
    if a.smoke:
        a.envs, a.steps, a.rollout = 2, 4000, 64
        a.eval_seeds, a.final_eval_seeds = 1, 1
    if a.steps < 1:
        ap.error("--steps 는 1 이상이어야 한다")
    if a.eval_every < 1:
        ap.error("--eval-every 는 1 이상이어야 한다(0 이면 while next_eval<=step 이 안 끊긴다)")

    log_path = os.path.join(a.out, "log.jsonl")
    if os.path.exists(log_path) or glob.glob(os.path.join(a.out, "ac-*.pt")):
        ap.error(f"`{a.out}` 에 이미 log.jsonl 또는 체크포인트가 있다 — 이어 쓰면 서로 다른 두"
                " 실행이 섞인다. 지우거나 다른 --out 을 써라.")
    os.makedirs(a.out, exist_ok=True)

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    stages = _stage_boards(a.curricula, a.smoke)

    cfg = PPOConfig()
    venv = None
    try:
        # venv 는 서브프로세스(비동기 벡터 환경)를 띄운다 — `--init` 오타 등으로 그 아래 어떤
        # 준비 코드가 죽어도 finally 가 반드시 타도록 venv 생성부터 이 try 안에 둔다.
        venv = make_vec_env(a.curricula, a.envs, EnvConfig(), seed=a.seed, asynchronous=not a.smoke)
        net = (ActorCritic.from_policy(a.init, device=dev) if a.init else ActorCritic()).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        buf = RolloutBuffer(a.rollout, a.envs, dev)
        gen = torch.Generator().manual_seed(a.seed)
        dagger_iter = (dagger_batches(load_dir(a.dagger_data), TrainConfig().batch_size,
                                      generator=gen, device=dev) if a.dagger_data else None)

        t0 = time.perf_counter()
        obs, _info = venv.reset(seed=a.seed)
        updates = 0
        prev_done = np.zeros(a.envs, dtype=bool)   # 직전 걸음에 끝난 환경 = 이번 걸음은 자동 리셋 더미
        step = 0
        next_eval = a.eval_every
        best_step, best_metric = None, None

        while step < a.steps:
            buf.reset()
            values_log, valid_log = [], []
            for _ in range(a.rollout):
                vec, objs, mask = (torch.as_tensor(x, device=dev) for x in vec_obs_to_arrays(obs))
                with torch.no_grad():
                    out = net.act(vec, objs, mask, generator=gen)
                action = {"control": out["control"].cpu().numpy(), "turn": out["turn"].cpu().numpy()}
                obs, reward, term, trunc, _info = venv.step(action)
                # 자동 리셋 더미 행(prev_done)은 reward=제 가치·done=1 로 줘서 GAE 사슬을 끊는다
                # (`_bootstrap_reward_done` 참고) — `done` 에 `term` 만 그대로 넣으면 truncated
                # 로 끝난 판 뒤에서 다음 판의 큰 가치가 (γλ)^k 로 새어 든다.
                reward_t, done_t = _bootstrap_reward_done(reward, term, prev_done, out["value"])
                valid_np = ~prev_done
                buf.add(vec=vec, objs=objs, mask=mask, raw=out["raw"], turn=out["turn"],
                        log_prob=out["log_prob"], value=out["value"],
                        reward=reward_t, done=done_t,
                        valid=torch.as_tensor(valid_np, dtype=torch.float32))
                values_log.append(out["value"].detach())
                valid_log.append(torch.as_tensor(valid_np, dtype=torch.bool, device=dev))
                prev_done = np.asarray(term) | np.asarray(trunc)
                step += a.envs
            with torch.no_grad():
                vec, objs, mask = (torch.as_tensor(x, device=dev) for x in vec_obs_to_arrays(obs))
                last_value = net.value(vec, objs, mask)
            buf.compute_gae(last_value, gamma=cfg.gamma, lam=cfg.lam)
            warming = updates < a.warmup_updates          # 가치만 먼저(스펙 §6.3)
            for p in net.policy.parameters():
                p.requires_grad_(not warming)
            stats = update(net, opt, buf, dagger_iter if not warming else None, cfg, step,
                           generator=gen)
            for p in net.policy.parameters():
                p.requires_grad_(True)
            updates += 1

            if step >= next_eval or step >= a.steps:
                # 진단(explained variance)도 `valid` 로 자동 리셋 더미 행을 뺀다 — 학습 배치에
                # 안 들어가는 행이 진단에는 섞이면 안 된다.
                values_t, valid_t = torch.stack(values_log), torch.stack(valid_log)
                ev_var = _explained_variance(buf.returns[valid_t], values_t[valid_t])
                net.eval()
                evs = _evaluate_stages(net.policy, stages, tuple(range(a.eval_seeds)))
                net.train()
                # `stats["updates"]` 는 이 롤아웃 안에서 도른 미니배치 최적화 걸음 수(ppo.update() 자체
                # 반환값)다 — 바깥 루프 반복 횟수(우리 `updates` 변수)와 이름이 겹치므로 `iter` 로 적는다.
                # 단계 결과는 "stages" 하위에 감싼다 — 커리큘럼 파일 이름을 그대로 키로 쓰면(예:
                # 누가 커리큘럼을 "policy.json" 처럼 지으면) 다른 통계 키와 겹칠 수 있다(`updates`
                # 이름 충돌로 한 번 데었다).
                row = {"step": step, "iter": updates, "elapsed_s": time.perf_counter() - t0,
                       **stats,
                       "explained_variance": ev_var if ev_var == ev_var else None,  # NaN → None(표준 JSON)
                       "log_std": net.policy.log_std.detach().cpu().tolist(),
                       "stages": {label: _public(ev) for label, ev in evs.items()}}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                net.save(os.path.join(a.out, f"ac-{step}.pt"))
                metric = _metric(evs)
                if best_metric is None or metric > best_metric:
                    best_metric, best_step = metric, step
                while next_eval <= step:
                    next_eval += a.eval_every
    finally:
        if venv is not None:
            venv.close()

    # 학습이 끝난 뒤 딱 한 번 전체 평가(--final-eval-seeds) — 이 값으로 ac-best.pt 를 최종
    # 선정하고 요약 JSON 을 채운다(주기 평가는 --eval-seeds 로 값싸게, 최종 선정만 신뢰도 있게).
    net.eval()
    final_seeds = tuple(range(a.final_eval_seeds))
    final_evs = _evaluate_stages(net.policy, stages, final_seeds)
    if best_step is not None and best_step != step:
        cand = ActorCritic.load(os.path.join(a.out, f"ac-{best_step}.pt"), device=dev)
        cand.eval()
        cand_evs = _evaluate_stages(cand.policy, stages, final_seeds)
        if _metric(cand_evs) > _metric(final_evs):
            cand.save(os.path.join(a.out, "ac-best.pt"))
            chosen_evs, chosen_step = cand_evs, best_step
        else:
            net.save(os.path.join(a.out, "ac-best.pt"))
            chosen_evs, chosen_step = final_evs, step
    else:
        net.save(os.path.join(a.out, "ac-best.pt"))
        chosen_evs, chosen_step = final_evs, step

    # `iters` = 바깥 루프(롤아웃) 반복 횟수 — log.jsonl 의 `updates`(ppo.update 내부 미니배치 수)와
    # 다른 뜻이라 요약 쪽은 이름을 갈랐다.
    summary = {"steps": step, "iters": updates, "seconds": time.perf_counter() - t0,
               "best_step": chosen_step,
               "stages": {label: _public(ev) for label, ev in chosen_evs.items()}}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
