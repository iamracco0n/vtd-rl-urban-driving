import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def test_성적표_스크립트는_한_판만_돌려도_된다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "run_m2b_env.py"),
                          "--boards", "course_H", "--out", str(tmp_path / "m2b.md")],
                         capture_output=True, text=True, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    text = open(tmp_path / "m2b.md", encoding="utf-8").read()
    assert "check_env" in text and "course_H" in text and "µs" in text
