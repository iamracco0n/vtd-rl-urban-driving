import importlib.util
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
PYTHON = os.path.join(REPO, ".venv", "bin", "python")
SWEEP = os.path.join(REPO, "scripts", "sweep_ppo.py")


def _env():
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)
    return e


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


def test__forbidden_flag가_seed와_out을_잡는다():
    """`--seed`/`--out`(등호 형태 포함)이 --seed-out 이 아니라 정확히 그 플래그일 때만 걸린다.

    `--out-root` 는 `--out` 으로 시작만 하지 `--out`/`--out=...` 그 자체는 아니므로 안 걸려야
    한다 — 아니면 스윕 스크립트 자신의 정상 인자까지 오검출한다.
    """
    module = _load_sweep_ppo_module()
    assert module._forbidden_flag(["--smoke", "--seed", "5"]) == "--seed"
    assert module._forbidden_flag(["--out=/tmp/x"]) == "--out"
    assert module._forbidden_flag(["--smoke", "--init", "x.pt"]) is None


def test_CLI에서_seed나_out을_뒤에_주면_거부한다():
    """실수로 다른 명령을 복붙해 `-- ... --seed 5` 가 섞이면 세 시드가 전부 같은 값으로

    도는데 에러가 안 나는 사고를 막는다(2026-09-21 리뷰 지적) — 학습을 하나도 안 띄우고
    인자 파싱 직후에 걸리므로 빠르다(`@pytest.mark.slow` 아님).
    """
    out = subprocess.run([PYTHON, SWEEP, "--out-root", "/tmp/불필요-존재안함",
                          "--name", "x", "--seeds", "0", "--", "--smoke", "--seed", "5"],
                         capture_output=True, text=True, env=_env(), cwd=REPO, timeout=30)
    assert out.returncode != 0
    assert "--seed" in out.stderr


@pytest.mark.slow
def test_시드_두_개를_돌리고_편차를_모은다(tmp_path):
    out = subprocess.run([PYTHON, SWEEP,
                          "--out-root", str(tmp_path / "sweep"), "--name", "smoke",
                          "--seeds", "0", "1", "--", "--smoke"],
                         capture_output=True, text=True, env=_env(), cwd=REPO, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["complete"] is True
    assert len(summary["runs"]) == 2
    assert {r["seed"] for r in summary["runs"]} == {0, 1}
    for stage in ("stage1", "stage2"):
        sp = summary["spread"][stage]["mean_score"]
        assert sp["min"] <= sp["mean"] <= sp["max"]
    assert os.path.exists(tmp_path / "sweep" / "sweep.json")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s0" / "log.jsonl")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s1" / "log.jsonl")
    # 시드마다 개별 요약도 즉시 남아야 한다(재개·부분 실패 복구의 근거).
    assert os.path.exists(tmp_path / "sweep" / "smoke-s0" / "summary.json")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s1" / "summary.json")


@pytest.mark.slow
def test_한_실행이_실패하면_멈춘다(tmp_path):
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
    out = subprocess.run([PYTHON, SWEEP,
                          "--out-root", str(tmp_path / "sweep"), "--name", "bad",
                          "--seeds", "0", "1", "--", "--smoke"],
                         capture_output=True, text=True, env=_env(), cwd=REPO, timeout=900)
    assert out.returncode != 0
    assert not os.path.exists(tmp_path / "sweep" / "bad-s1")


@pytest.mark.slow
def test_둘째_시드가_실패해도_첫_시드_결과는_남고_resume이_건너뛴다(tmp_path):
    """리뷰 지적(2026-09-21, Critical): 원래는 루프가 끝까지 성공한 뒤에만 sweep.json 을

    썼다 — 9 개 중 8 번째가 죽으면 앞선 7 개의 cross-seed 스프레드가 파이썬 지역 변수 속에서만
    살다 통째로 사라졌다. 여기서는 (첫 시드가 아니라) **둘째** 시드를 실패시켜 그 버그가
    건드리는 경로(루프 중간에 멈춤)를 직접 겨눈다 — 첫 시드가 실패하는 경로는
    `test_한_실행이_실패하면_멈춘다` 가 이미 다룬다.
    """
    sweep_dir = tmp_path / "sweep"
    # 시드 1 폴더에만 재사용 가드 걸림돌을 심는다 — 시드 0 은 끝까지 성공해야 "부분 실패"가 된다.
    bad_s1 = sweep_dir / "bad-s1"
    bad_s1.mkdir(parents=True)
    poison = bad_s1 / "log.jsonl"
    poison.write_text("")

    base = [PYTHON, SWEEP, "--out-root", str(sweep_dir), "--name", "bad", "--seeds", "0", "1"]
    out1 = subprocess.run(base + ["--", "--smoke"], capture_output=True, text=True,
                          env=_env(), cwd=REPO, timeout=900)
    assert out1.returncode != 0

    # ① 첫 시드(성공)의 결과가 개별 파일로도, 부분 sweep.json 으로도 남아 있어야 한다.
    s0_summary_path = sweep_dir / "bad-s0" / "summary.json"
    assert s0_summary_path.exists()
    with open(sweep_dir / "sweep.json", encoding="utf-8") as f:
        partial = json.load(f)
    assert partial["complete"] is False
    assert [r["seed"] for r in partial["runs"]] == [0]

    # ② --resume 으로 다시 돌리면 시드 0 은 건너뛰어야 한다. 시드 1 의 실패 원인(가짜
    # log.jsonl)을 지워 이번엔 진짜로 성공하게 만든다 — 만약 --resume 이 시드 0 을 다시
    # 부르려 한다면, 방금 실제 학습으로 생긴 진짜 log.jsonl 에 재사용 가드가 걸려 rc!=0 이
    # 될 것이다(즉 이 assert 자체가 "시드 0 을 건너뛰었다"의 증거다).
    poison.unlink()
    out2 = subprocess.run(base + ["--resume", "--", "--smoke"], capture_output=True, text=True,
                          env=_env(), cwd=REPO, timeout=900)
    assert out2.returncode == 0, out2.stderr[-3000:]
    final = json.loads(out2.stdout.strip().splitlines()[-1])
    assert final["complete"] is True
    assert {r["seed"] for r in final["runs"]} == {0, 1}
    # 시드 0 의 요약은 (재실행이 아니라) 저장된 파일에서 그대로 왔어야 한다.
    with open(s0_summary_path, encoding="utf-8") as f:
        saved_s0 = json.load(f)
    reused = next(r["summary"] for r in final["runs"] if r["seed"] == 0)
    assert reused == saved_s0
