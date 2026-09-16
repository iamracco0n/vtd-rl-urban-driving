import os
import subprocess

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def run_script(*args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                           os.path.join(REPO, "scripts", "run_m2b_env.py"), *args],
                          capture_output=True, text=True, env=env, cwd=REPO)


def test_성적표_스크립트는_한_판만_돌려도_된다(tmp_path):
    out = run_script("--boards", "course_H", "--out", str(tmp_path / "m2b.md"))
    assert out.returncode == 0, out.stderr[-2000:]
    text = open(tmp_path / "m2b.md", encoding="utf-8").read()
    assert "check_env" in text and "course_H" in text and "µs" in text


def test_없는_판_이름이면_0_이_아니다(tmp_path):
    """--boards 오타로 아무 판도 안 골리면 빈 성적표를 0 으로 내지 않는다."""
    out = run_script("--boards", "course_없음", "--out", str(tmp_path / "m2b.md"))
    assert out.returncode != 0
    assert not (tmp_path / "m2b.md").exists()
