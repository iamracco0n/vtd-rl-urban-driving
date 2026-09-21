import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


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
    assert summary["steps"] > 0 and "stage1" in summary
    assert os.path.exists(tmp_path / "run" / "log.jsonl")
    assert os.path.exists(tmp_path / "run" / "ac-best.pt")


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
