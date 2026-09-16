import json
import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def test_연습_모드는_한_바퀴를_끝낸다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "run_dagger.py"),
                          "--smoke", "--out", str(tmp_path / "run"),
                          "--report", str(tmp_path / "m3.md")],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["rounds"] == 1 and summary["samples"] > 0
    assert os.path.exists(tmp_path / "run" / "policy-r0.pt")
    assert os.path.exists(tmp_path / "run" / "log.jsonl")
    text = open(tmp_path / "m3.md", encoding="utf-8").read()
    assert "DAgger" in text and "완주율" in text
