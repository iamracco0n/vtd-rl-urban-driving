import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_run_dagger_module():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다."""
    path = os.path.join(REPO, "scripts", "run_dagger.py")
    spec = importlib.util.spec_from_file_location("run_dagger_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_shard(n_stopped_go):
    """n_stopped_go 개는 '정지+선생님 출발 지시', 나머지 1개는 아닌 표본."""
    from vtd_rl.policy.dataset import Shard
    n = n_stopped_go + 1
    vec = np.zeros((n, 4), dtype=np.float32)
    vec[:n_stopped_go, 0] = 0.0     # 정지(ego 속도 0)
    vec[n_stopped_go:, 0] = 0.9     # 주행 중
    control = np.zeros((n, 2), dtype=np.float32)
    control[:n_stopped_go, 1] = 0.5     # 선생님이 출발하라 함(양의 가속)
    control[n_stopped_go:, 1] = -0.3    # 그 외는 감속
    return Shard(vec=vec, objs=np.zeros((n, 1, 12), dtype=np.float32),
                mask=np.zeros((n, 1), dtype=np.float32), control=control,
                turn=np.zeros(n, dtype=np.int64), meta={})


def test_stall_start_count_는_라운드마다_다른_시드_파일명을_찾는다(tmp_path):
    """라운드 1 의 시드는 s 가 아니라 1000*rnd+s 다 — 라운드 0 만으로는 이 버그가 안 보인다

    (1000*0+s == s 라서 우연히 맞아떨어진다). 실제로 이 형태의 버그를 겪었다.
    """
    from vtd_rl.policy.dataset import save_shard

    run_dagger = _load_run_dagger_module()
    data_dir = str(tmp_path)
    names = ["course_A"]
    num_seeds = 2

    save_shard(_make_shard(1), os.path.join(data_dir, "r0-course_A-s0.npz"))
    save_shard(_make_shard(2), os.path.join(data_dir, "r0-course_A-s1.npz"))
    # 라운드 1: 실제 파일명은 시드 1000*1+s = 1000, 1001 이다.
    save_shard(_make_shard(0), os.path.join(data_dir, "r1-course_A-s1000.npz"))
    save_shard(_make_shard(3), os.path.join(data_dir, "r1-course_A-s1001.npz"))

    assert run_dagger._stall_start_count(data_dir, 0, names, num_seeds) == 1 + 2
    assert run_dagger._stall_start_count(data_dir, 1, names, num_seeds) == 0 + 3


@pytest.mark.slow
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
