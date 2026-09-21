import json
import os
import subprocess

import pytest

from vtd_rl.policy.evaluate import EpisodeOutcome, violation_counts

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
PYTHON = os.path.join(REPO, ".venv", "bin", "python")


def _env():
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)
    return e


def _train_smoke(out_dir: str, *extra: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, os.path.join(REPO, "scripts", "train_ppo.py"),
                           "--smoke", "--out", out_dir, "--seed", "0", *extra],
                          capture_output=True, text=True, env=_env(), cwd=REPO, timeout=timeout)


def _report(args: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, os.path.join(REPO, "scripts", "report_m4a.py"), *args],
                          capture_output=True, text=True, env=_env(), cwd=REPO, timeout=timeout)


def test_위반_세기는_구간_슬롯_단위():
    # 감점표의 항목 키는 **정수**다(score_fma.ITEMS 는 1~15 의 int).
    ev = {"episodes": [
        EpisodeOutcome("A", 0, "goal", 10, 0.0, 90.0,
                       [{7: "major"}, {7: "major", 2: "minor"}, {}]),
        EpisodeOutcome("B", 0, "goal", 10, 0.0, 95.0, [{2: "minor"}]),
    ]}
    counts = violation_counts(ev)
    assert counts[7] == {"minor": 0, "major": 2}
    assert counts[2] == {"minor": 2, "major": 0}


def test_M3_스크립트도_같은_함수를_쓴다():
    src = open(os.path.join(REPO, "scripts", "run_dagger.py"), encoding="utf-8").read()
    assert "from vtd_rl.policy.evaluate import" in src and "violation_counts" in src
    assert "def _violation_counts" not in src          # 복사본을 남기지 않았다


@pytest.mark.slow
def test_성적표가_필수_항목을_담는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    run_dir = str(tmp_path / "run")
    train = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                            os.path.join(REPO, "scripts", "train_ppo.py"),
                            "--smoke", "--out", run_dir, "--seed", "0"],
                           capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert train.returncode == 0, train.stderr[-3000:]
    report = str(tmp_path / "m4a.md")
    made = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                           os.path.join(REPO, "scripts", "report_m4a.py"),
                           "--run", run_dir, "--out", report, "--skip-eval"],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert made.returncode == 0, made.stderr[-3000:]
    text = open(report, encoding="utf-8").read()
    for needed in ("PPO", "완주율", "항목별 위반", "목표", "시드", "커리큘럼"):
        assert needed in text, needed
    assert json.loads(open(os.path.join(run_dir, "log.jsonl"),
                           encoding="utf-8").readline())["step"] >= 0


@pytest.mark.slow
def test_compare_는_실행마다_다른_실측값을_보여준다(tmp_path):
    """"인자가 파싱된다" 만 보는 테스트는 의미가 없다 — 두 실행이 실제로 다른 하이퍼파라미터로

    돌았을 때 성적표가 그 차이를 실측값으로(손으로 짐작한 문자열이 아니라) 보여주는지 잠근다.
    """
    run_a, run_b = str(tmp_path / "run_a"), str(tmp_path / "run_b")
    assert _train_smoke(run_a).returncode == 0
    train_b = _train_smoke(run_b, "--entropy-coef", "0.05")
    assert train_b.returncode == 0, train_b.stderr[-3000:]

    report = str(tmp_path / "cmp.md")
    made = _report(["--run", run_a, "--compare", run_b, "--out", report, "--skip-eval"])
    assert made.returncode == 0, made.stderr[-3000:]
    text = open(report, encoding="utf-8").read()

    # 두 실행 이름이 다 나온다
    assert os.path.basename(run_a) in text
    assert os.path.basename(run_b) in text
    # run_b 만 엔트로피 계수를 바꿨다 — 그 실측값이 실제로 다르게 나와야 한다(문자열을 흉내낸
    # 표가 아니라 진짜 log.jsonl 의 hparams 를 읽었다는 증거).
    assert "entropy_coef=0.05" in text
    assert "(기본값)" in text          # run_a 는 바뀐 게 없다
    # best_step 은 체크포인트 파일을 실제로 비교해 되찾은 값이다 — 스모크는 평가가 한 번뿐이라
    # log.jsonl 마지막 줄의 스텝과 같아야 한다.
    with open(os.path.join(run_a, "log.jsonl"), encoding="utf-8") as f:
        last_step_a = json.loads(list(f)[-1])["step"]
    with open(os.path.join(run_b, "log.jsonl"), encoding="utf-8") as f:
        last_step_b = json.loads(list(f)[-1])["step"]
    assert str(last_step_a) in text
    assert str(last_step_b) in text
    # 학습 곡선 표도 실행마다 나란히 나온다
    assert text.count("학습 곡선") >= 2


@pytest.mark.slow
def test_notes_파일_내용이_그대로_들어가고_안_주면_정직한_한_줄만_남는다(tmp_path):
    """`--notes` 는 내용을 검증·가공하지 않고 그대로 삽입만 한다 — 그 대신 안 주면 지어낸

    자리표시자가 아니라 정확히 지시된 한 줄("--notes 로 주지 않았다")만 남아야 한다.
    """
    run_dir = str(tmp_path / "run")
    assert _train_smoke(run_dir).returncode == 0

    marker = "엔트로피 붕괴가 원인이다 — 조향 log_std 가 하한 -2.000 에 붙박였다(마커-e7f3a2)."
    notes_path = tmp_path / "notes.md"
    notes_path.write_text(f"### 관찰\n\n{marker}\n", encoding="utf-8")

    with_notes = str(tmp_path / "with_notes.md")
    made1 = _report(["--run", run_dir, "--out", with_notes, "--skip-eval", "--notes", str(notes_path)])
    assert made1.returncode == 0, made1.stderr[-3000:]
    text1 = open(with_notes, encoding="utf-8").read()
    assert marker in text1
    assert "관찰과 해석 — `--notes` 로 주지 않았다" not in text1

    without_notes = str(tmp_path / "without_notes.md")
    made2 = _report(["--run", run_dir, "--out", without_notes, "--skip-eval"])
    assert made2.returncode == 0, made2.stderr[-3000:]
    text2 = open(without_notes, encoding="utf-8").read()
    assert "관찰과 해석 — `--notes` 로 주지 않았다" in text2
    assert marker not in text2
