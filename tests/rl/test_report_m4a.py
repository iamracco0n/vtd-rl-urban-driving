import importlib.util
import json
import os
import subprocess

import pytest

from vtd_rl.policy.evaluate import EpisodeOutcome, violation_counts

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
PYTHON = os.path.join(REPO, ".venv", "bin", "python")


def _load_report_m4a_module():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다

    (`tests/rl/test_train_ppo.py::_load_train_ppo_module` 과 같은 패턴). private 헬퍼
    (`_train_seed_line`·`_curve_table`·`_item_goal_line`·`_goal_lines`) 를 서브프로세스 없이
    직접 불러 잠글 때 쓴다.
    """
    path = os.path.join(REPO, "scripts", "report_m4a.py")
    spec = importlib.util.spec_from_file_location("report_m4a_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env():
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)
    return e


def _write_minimal_log(run_dir: str, hparams=None):
    """`--skip-eval` 경로만 확인할 때 쓰는, 실제 스키마를 그대로 흉내 낸 한 줄짜리 log.jsonl.

    학습을 실제로 돌리지 않아 빠르다 — 실제 필드 이름·모양은 `scripts/train_ppo.py` 가 쓰는
    것과 정확히 같다(2026-09-21 실측한 실물 스키마).
    """
    os.makedirs(run_dir, exist_ok=True)
    row = {"step": 100, "iter": 1, "elapsed_s": 1.23, "policy": 0.0, "value": 0.0, "entropy": 0.0,
           "approx_kl": 0.0, "clip_frac": 0.0, "imitation": 0.0, "imitation_coef": 1.0, "updates": 1,
           "explained_variance": 0.5, "log_std": [-1.0, -1.0],
           "stages": {"stage1": {"goal_rate": 1.0, "mean_score": 90.0, "mean_score_raw": 90.0,
                                 "mean_reward": 102.5},
                      "stage2": {"goal_rate": 0.5, "mean_score": 45.0, "mean_score_raw": 45.0,
                                 "mean_reward": 22.5}}}
    if hparams is not None:
        row["hparams"] = hparams
    with open(os.path.join(run_dir, "log.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def test_학습_시드는_hparams에서_읽고_없으면_train_seed로만_폴백한다(tmp_path):
    """2026-09-22 리뷰(Important #1): `hparams["seed"]` 가 있는데도 "기록 안 됨" 이라 말하면 안

    된다 — `train_ppo.py` 가 실제로 쓴 값을 성적표 자신이 없다고 말하고 사람에게 손으로
    다시 넣으라고 시키는 꼴이다. hparams 가 없는(로깅 이전) 실행에서만 `--train-seed` 폴백.
    """
    module = _load_report_m4a_module()
    # 헬퍼 자체의 진리표 — 서브프로세스 없이 직접
    assert module._train_seed_line({"seed": 0}, None) == "0(log.jsonl 의 hparams 에서 읽음)"
    assert "직접 입력값" in module._train_seed_line(None, 7)
    assert module._train_seed_line(None, None) == (
        "기록 안 됨(log.jsonl 에 hparams 가 없고 --train-seed 도 안 줬다)")
    assert module._train_seed_line({"seed": 3}, 7) == "3(log.jsonl 의 hparams 에서 읽음)"  # hparams 우선

    # 배선 — main() 이 이 헬퍼를 실제로 불러 성적표에 싣는지 CLI 왕복으로 확인(가짜 log.jsonl,
    # 학습을 실제로 안 돌려 빠르다)
    run_with, run_without = str(tmp_path / "run_with"), str(tmp_path / "run_without")
    _write_minimal_log(run_with, {"seed": 0, "lr": 3e-4, "entropy_coef": 0.005,
                                  "target_kl": 0.03, "imitation_half_life": 2_000_000})
    _write_minimal_log(run_without, None)

    out1 = str(tmp_path / "with.md")
    made1 = _report(["--run", run_with, "--out", out1, "--skip-eval"])
    assert made1.returncode == 0, made1.stderr[-3000:]
    text1 = open(out1, encoding="utf-8").read()
    assert "학습 시드: 0(log.jsonl 의 hparams 에서 읽음)" in text1
    assert "기록 안 됨" not in text1

    out2 = str(tmp_path / "without.md")
    made2 = _report(["--run", run_without, "--out", out2, "--skip-eval"])
    assert made2.returncode == 0, made2.stderr[-3000:]
    assert "학습 시드: 기록 안 됨" in open(out2, encoding="utf-8").read()

    out3 = str(tmp_path / "fallback.md")
    made3 = _report(["--run", run_without, "--out", out3, "--skip-eval", "--train-seed", "7"])
    assert made3.returncode == 0, made3.stderr[-3000:]
    assert "학습 시드: 7(직접 입력값" in open(out3, encoding="utf-8").read()


def test_학습_곡선_표에_평균_보상_열이_있다():
    """2026-09-22 리뷰(Important #2): 계획서가 명시한 평균 보상 열이 빠져 있었다 — 관찰 노트가

    인용하는 숫자(단계① 102.5 → 22.5)를 성적표에서 확인할 수 있어야 한다.
    """
    module = _load_report_m4a_module()
    rows = [{"step": 100, "iter": 1, "explained_variance": 0.5, "log_std": [-1.0, -1.0],
             "imitation_coef": 1.0, "approx_kl": 0.01,
             "stages": {"stage1": {"goal_rate": 1.0, "mean_score": 90.0, "mean_reward": 102.5},
                        "stage2": {"goal_rate": 0.5, "mean_score": 45.0, "mean_reward": 22.5}}}]
    text = "\n".join(module._curve_table(rows, ["stage1", "stage2"]))
    assert "평균 보상" in text
    assert "102.5" in text and "22.5" in text


def test_항목_문턱은_시드3개_합계_기준이라_다른_시드수면_판정을_보류한다():
    """2026-09-22 리뷰(Important #3): `ITEM7_MAJOR_MAX`/`ITEM2_MAJOR_MAX` 는 시드 3개 합계

    전제인데 결합이 없었다 — `--eval-seeds` 를 바꾸면 판정이 조용히 뒤집힌다(단계①은 시드
    무관 같은 판 6개라 건수가 시드 수에 정확히 비례한다). 시드 3 이 아니면 문턱을 스케일하지
    않고 보류해야 한다.
    """
    module = _load_report_m4a_module()

    def ev(goal_rate, mean_score, item, major_count):
        sheet = [{item: "major"} for _ in range(major_count)]
        return {"goal_rate": goal_rate, "mean_score": mean_score,
                "episodes": [EpisodeOutcome("X", 0, "goal", 10, 0.0, mean_score, sheet)]}

    teacher = {"goal_rate": 1.0, "mean_score": 99.0, "episodes": []}
    teacher_ev = {module.STAGE1_LABEL: teacher, module.STAGE2_LABEL: teacher}

    # 시드 3개(문턱이 전제하는 그 조건)일 때는 정상적으로 달성/미달을 낸다.
    m4a_ev_3 = {module.STAGE1_LABEL: ev(1.0, 95.0, 2, 4),    # 문턱 4 이하 → 달성
                module.STAGE2_LABEL: ev(1.0, 95.0, 7, 20)}   # 문턱 19 초과 → 미달
    text3 = "\n".join(module._goal_lines(m4a_ev_3, teacher_ev, rows=[], best_step=None, eval_seeds=3))
    assert "보류" not in text3
    assert "목표 4 줄 중 **3 줄 달성**." in text3
    assert "항목②" in text3 and "달성" in text3
    assert "항목⑦" in text3 and "미달" in text3

    # 시드 1개로 돌리면 — 같은 절대 건수라도(오히려 더 적은데도) 문턱을 1/3로 스케일하지 않고
    # 두 줄 다 보류한다.
    m4a_ev_1 = {module.STAGE1_LABEL: ev(1.0, 95.0, 2, 4), module.STAGE2_LABEL: ev(1.0, 95.0, 7, 1)}
    text1 = "\n".join(module._goal_lines(m4a_ev_1, teacher_ev, rows=[], best_step=None, eval_seeds=1))
    assert text1.count("보류") >= 2
    assert "목표 4 줄 중 **2 줄 달성**(2 줄 보류" in text1
    assert "달성" not in text1.split("항목②")[1].split("\n")[0]   # 항목② 줄 자체엔 "달성" 이 없다(보류)
    assert "미달" not in text1.split("항목⑦")[1].split("\n")[0]   # 항목⑦ 줄 자체엔 "미달" 이 없다(보류)
