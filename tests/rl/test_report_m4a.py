import json
import os
import subprocess

import pytest

from vtd_rl.policy.evaluate import EpisodeOutcome, violation_counts

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


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
