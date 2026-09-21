import importlib.util
import json
import math
import os
import subprocess

import pytest
import torch

from vtd_rl.rl.actor_critic import ActorCritic
from vtd_rl.rl.buffer import RolloutBuffer
from vtd_rl.rl.ppo import PPOConfig

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_train_ppo_module():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다

    (`tests/policy/test_dagger_loop.py` 와 같은 패턴).
    """
    path = os.path.join(REPO, "scripts", "train_ppo.py")
    spec = importlib.util.spec_from_file_location("train_ppo_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_인자를_안_주면_PPOConfig_기본값과_같다():
    """`--entropy-coef`/`--lr`/`--target-kl`/`--imitation-half-life` 를 하나도 안 주면

    `_build_cfg` 가 만드는 `cfg` 는 `PPOConfig()` 와 완전히 같아야 한다 — CLI 로 여는 것
    자체가 기본 동작을 바꾸면 안 된다(2026-09-21 지시: "아무 인자도 안 주면 동작이 지금과
    100% 같아야 한다").
    """
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    cfg = module._build_cfg(a)
    assert cfg == PPOConfig()


def test_하이퍼파라미터_인자가_cfg와_옵티마이저에_실제로_반영된다():
    """"인자가 파싱된다"만 보는 공허한 테스트가 되지 않도록, 실제 `torch.optim.Adam` 을 만들어

    `opt.param_groups[0]["lr"]` 을 직접 본다 — 이 프로젝트에서 `PPOConfig.gamma`/`lam` 이
    `ppo.update()` 에 안 쓰여 조용히 무시된 적이 있어(`compute_gae` 호출부), `--lr` 도 같은
    함정(옵티마이저가 하드코딩된 값을 쓰는)에 빠질 수 있다는 게 코디네이터의 우려였다.
    """
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args([
        "--out", "/tmp/불필요-존재안함", "--lr", "0.00013", "--entropy-coef", "0.2",
        "--target-kl", "0.5", "--imitation-half-life", "12345"])
    cfg = module._build_cfg(a)
    assert cfg.lr == 0.00013
    assert cfg.entropy_coef == 0.2
    assert cfg.target_kl == 0.5
    assert cfg.imitation_half_life == 12345
    # 안 건드린 필드는 그대로(네 필드만 골라 바꿨다는 확인).
    default_cfg = PPOConfig()
    assert cfg.clip == default_cfg.clip and cfg.gamma == default_cfg.gamma
    assert cfg.lam == default_cfg.lam and cfg.value_coef == default_cfg.value_coef

    # main() 이 실제로 쓰는 그 함수로 옵티마이저를 만들어 lr 이 진짜로 닿는지 본다.
    net = module.ActorCritic()
    opt = module._build_optimizer(net, cfg)
    assert opt.param_groups[0]["lr"] == 0.00013


def test_자동_리셋_더미_행이_GAE_사슬을_끊는다():
    """2026-09-21 리뷰 재현 — truncated(시간 초과·정체)로 끝난 판 뒤 자동 리셋 더미 행의

    `done` 에 `term` 만(거짓) 넣으면 다음 판의 큰 가치가 `(γλ)^k` 로 새어 들어와 이전 진짜
    걸음들의 이점을 체계적으로 부풀린다. `_bootstrap_reward_done` 이 더미 행의 reward 를
    제 가치로, done 을 1 로 바꿔야 그 자리에서 사슬이 끊긴다.

    시나리오(리뷰가 직접 재현한 수치와 동일): 4 걸음 — 0,1 은 truncated 로 끝나는 판의 진짜
    전이(마지막 실제 관측의 가치 V(final)=1 은 더미 행 자신의 값으로 나타난다), 2 는 그 뒤에
    오는 자동 리셋 더미 행, 3 은 새 판의 진짜 첫 전이(V(new)=50). γ=0.99, λ=0.95.
    고치기 전: [3.642, 2.82, 1.945, -49.5]. 고친 뒤: [1.921, 0.99, 0.0, -49.5]
    (마지막 값 -49.5 는 더미 행보다 뒤(시간순으로는 앞서 처리되는) 행이라 그대로다 — 이점이
    거꾸로만 샌다는 것 자체도 이 값으로 확인된다).
    """
    module = _load_train_ppo_module()

    def make_buf(reward2, done2):
        buf = RolloutBuffer(4, 1, torch.device("cpu"))

        def row(reward, value, done):
            n = 1
            buf.add(vec=torch.zeros(n, 1), objs=torch.zeros(n, 1, 1), mask=torch.zeros(n, 1),
                    raw=torch.zeros(n, 1), turn=torch.zeros(n, dtype=torch.long),
                    log_prob=torch.zeros(n), value=torch.full((n,), float(value)),
                    reward=torch.full((n,), float(reward)), done=torch.full((n,), float(done)),
                    valid=torch.ones(n))

        row(0.99, 0.0, False)     # 0: 진짜 전이
        row(0.0, 0.0, False)      # 1: truncated 로 끝나는 마지막 진짜 전이(term=False)
        row(reward2, 1.0, done2)  # 2: 자동 리셋 더미 행 — V(final)=1
        row(0.5, 50.0, False)     # 3: 새 판의 진짜 첫 전이 — V(new)=50
        buf.compute_gae(last_value=torch.zeros(1), gamma=0.99, lam=0.95)
        return [round(x, 3) for x in buf.advantages_raw.reshape(-1).tolist()]

    # RED 증거: done 에 term 만(거짓) 넣은 옛 방식 — 사슬이 안 끊긴다.
    buggy = make_buf(reward2=0.0, done2=False)
    assert buggy == [3.642, 2.82, 1.945, -49.5]

    # 실제 프로덕션 함수로 더미 행의 reward·done 을 만든다 — 이 함수를 빼거나 원래대로
    # 되돌리면(reward=term 그대로) 위 buggy 값이 나와 아래 단언이 깨진다.
    reward_t, done_t = module._bootstrap_reward_done(
        reward=[0.0], term=[False], prev_done=[True], value=torch.tensor([1.0]))
    fixed = make_buf(reward2=float(reward_t.item()), done2=float(done_t.item()))
    assert fixed == [1.921, 0.99, 0.0, -49.5]


@pytest.mark.slow
def test_연습_모드가_한_바퀴를_끝낸다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["steps"] > 0 and "stages" in summary and "stage1" in summary["stages"]
    log_path = tmp_path / "run" / "log.jsonl"
    best_path = tmp_path / "run" / "ac-best.pt"
    assert os.path.exists(log_path)
    assert os.path.exists(best_path)

    with open(log_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    last_row = json.loads(lines[-1])
    assert len(last_row["log_std"]) == 2
    assert math.isfinite(last_row["explained_variance"])
    assert "stage1" in last_row["stages"]
    # 이번 실행에 쓴 하이퍼파라미터가 산출물(log.jsonl·요약 JSON)에 실제로 남는지.
    assert last_row["hparams"]["lr"] == pytest.approx(0.0003)
    assert "hparams" in summary and summary["hparams"]["entropy_coef"] == pytest.approx(0.005)

    ActorCritic.load(str(best_path))   # ac-best.pt 가 실제로 ActorCritic 으로 읽혀야 한다


@pytest.mark.slow
def test_같은_폴더에_두_번_쓰지_않는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    args = [os.path.join(REPO, ".venv", "bin", "python"),
            os.path.join(REPO, "scripts", "train_ppo.py"), "--smoke",
            "--out", str(tmp_path / "run")]
    first = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert first.returncode == 0
    second = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert second.returncode != 0 and "이미" in (second.stderr + second.stdout)


@pytest.mark.slow
def test_연습_모드가_계측과_판정을_남긴다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                          "--imitation-sigma", "detach"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])

    assert summary["hparams"]["imitation_sigma"] == "detach"
    assert len(summary["verdict"]) == 4
    assert {v["name"] for v in summary["verdict"]} == {
        "완주율", "선생님 대비 점수", "출발점 대비 점수", "전 항목 중대 위반"}
    assert all(isinstance(v["ok"], bool) and v["line"] for v in summary["verdict"])

    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl", encoding="utf-8")]
    assert rows, "log.jsonl 이 비어 있다"
    for r in rows:
        for k in ("rollout_return_mean", "rollout_return_n", "rollout_len_mean",
                  "drift_l2", "drift_rel", "drift_log_std", "drift_rest_rel"):
            assert k in r, k
    assert any(r["rollout_return_n"] > 0 for r in rows), "끝난 판이 한 번도 안 잡혔다"
    # 드리프트는 처음엔 0 에 가깝고 학습이 돌수록 커진다. `drift_log_std` 를 `drift_l2` 와
    # 따로 확인한다 — `drift_l2`(전체망 노름)만 보면 `drift_log_std` 를 로그에서 빼거나 항상
    # 0 으로 찍어도 못 잡는다(trunk 수만 파라미터에 묻힌다, diagnostics.py `policy_drift` 참고
    # — M4b 설계에서 "가장 중요하다"고 지목한 값이라 별도 단언을 둔다).
    assert rows[0]["drift_l2"] < rows[-1]["drift_l2"]
    assert rows[0]["drift_log_std"] < rows[-1]["drift_log_std"]
    evals = [r for r in rows if "stages" in r]
    assert evals and any(k.startswith("outcome_") for k in evals[-1])


def test_시그마_모드가_cfg에_닿는다():
    mod = _load_train_ppo_module()   # 이 파일이 이미 쓰는 importlib 헬퍼(이름 그대로)
    a = mod._build_parser().parse_args(["--out", "x", "--imitation-sigma", "detach"])
    assert mod._build_cfg(a).imitation_sigma == "detach"
    b = mod._build_parser().parse_args(["--out", "x"])
    assert mod._build_cfg(b).imitation_sigma == "learn"
