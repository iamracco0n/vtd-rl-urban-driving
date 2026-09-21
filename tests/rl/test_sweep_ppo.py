import importlib.util
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_sweep_ppo_module():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다

    (`tests/rl/test_train_ppo.py::_load_train_ppo_module` 과 같은 패턴). 서브프로세스 없이
    순수 함수 `_spread` 를 직접 잠글 때 쓴다.
    """
    path = os.path.join(REPO, "scripts", "sweep_ppo.py")
    spec = importlib.util.spec_from_file_location("sweep_ppo_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test__spread_는_min_max_mean을_바르게_가른다():
    """스모크 학습은 `mean_score` 가 시드마다 0.0 으로 겹치기 쉬워(단계를 못 끝내면 0점)

    실제 학습을 낀 e2e 테스트만으로는 `min`/`max` 가 뒤바뀌어도 (동률이라) 못 잡는다
    (2026-09-21 돌연변이 실험 실측 — 두 시드 스모크 러너 결과가 우연히 동률이라 값이 겹쳤다).
    그래서 값이 뚜렷이 다른 합성 입력으로 `_spread` 자체를 직접 잠근다.
    """
    module = _load_sweep_ppo_module()
    assert module._spread([1.0, 5.0, 3.0]) == {"min": 1.0, "max": 5.0, "mean": 3.0}


@pytest.mark.slow
def test_시드_두_개를_돌리고_편차를_모은다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "sweep_ppo.py"),
                          "--out-root", str(tmp_path / "sweep"), "--name", "smoke",
                          "--seeds", "0", "1", "--", "--smoke"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(summary["runs"]) == 2
    assert {r["seed"] for r in summary["runs"]} == {0, 1}
    for stage in ("stage1", "stage2"):
        sp = summary["spread"][stage]["mean_score"]
        assert sp["min"] <= sp["mean"] <= sp["max"]
    assert os.path.exists(tmp_path / "sweep" / "sweep.json")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s0" / "log.jsonl")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s1" / "log.jsonl")


@pytest.mark.slow
def test_한_실행이_실패하면_멈춘다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 시드 0 폴더에 log.jsonl 을 미리 심어 train_ppo.py 의 재사용 가드(같은 --out 에 이미
    # log.jsonl 이 있으면 거부)가 걷어차게 만든다 — 진짜 "멈췄는지"를 가르려면 시드 0 의
    # 실패가 뒷자리에 아무 흔적도 안 남기는 실패면 안 된다. 시드 1 에는 정상적인 --smoke 를
    # 주므로, 러너가 실제로 멈추지 않고 계속 돌면 시드 1 은 그냥 성공해 smoke-s1/log.jsonl 이
    # 생겨 버린다(원안의 `--steps 0` 은 train_ppo.py 가 --out 을 만들기도 전에 거부해 '멈춤'과
    # '계속하다 이후에 죽음'을 못 갈랐다 — 두 경우 다 시드 1 폴더가 안 생겼다, 2026-09-21 실측
    # 확인: `continue` 로 바꿔도 이 조합으로는 테스트가 그대로 통과해 버렸다).
    bad_s0 = tmp_path / "sweep" / "bad-s0"
    bad_s0.mkdir(parents=True)
    (bad_s0 / "log.jsonl").write_text("")
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "sweep_ppo.py"),
                          "--out-root", str(tmp_path / "sweep"), "--name", "bad",
                          "--seeds", "0", "1", "--", "--smoke"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert out.returncode != 0
    assert not os.path.exists(tmp_path / "sweep" / "bad-s1")
