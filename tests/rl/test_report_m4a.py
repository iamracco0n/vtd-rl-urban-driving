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


def test_항목_문턱_상수가_사라졌다():
    """Task 6 정본: 항목별 절대 건수 문턱은 `verdict.judge()` 의 총점·중대 합계 판정으로

    대체됐다 — 복사본을 남기지 않는다.
    """
    src = open(os.path.join(REPO, "scripts", "report_m4a.py"), encoding="utf-8").read()
    assert "ITEM7_MAJOR_MAX" not in src and "ITEM2_MAJOR_MAX" not in src
    assert "_item_goal_line" not in src
    assert "from vtd_rl.eval.verdict import" in src


def test_헤드라인은_전제를_빼고_실질_판정만_센다():
    """`judge()` 가 낸 네 줄 중 완주율·선생님 대비 점수(`precondition=True`)는 3번(출발점 대비

    점수)에 수학적으로 포함된 전제다 — 헤드라인이 그 둘까지 세면 M4a 를 망친 "형식상 N 줄
    달성" 부풀림을 재현한다. 실질 판정은 항상 2 줄(출발점 대비 점수·전 항목 중대 위반)뿐이다.
    """
    module = _load_report_m4a_module()

    def ev(goal_rate, mean_score):
        return {"goal_rate": goal_rate, "mean_score": mean_score, "episodes": []}

    # 두 실질 판정 줄(출발점 대비 점수·중대 위반)이 모두 통과하도록 M3 문턱을 넉넉히 넘긴다.
    m4a_ev = {module.STAGE1_LABEL: ev(1.0, 99.6), module.STAGE2_LABEL: ev(1.0, 95.0)}
    teacher_ev = {module.STAGE1_LABEL: ev(1.0, 99.0), module.STAGE2_LABEL: ev(1.0, 96.0)}
    major_totals = {module.STAGE1_LABEL: 0, module.STAGE2_LABEL: 0}
    text = "\n".join(module._goal_lines(m4a_ev, teacher_ev, major_totals, rows=[], best_step=None,
                                        eval_seeds=3))
    assert "실질 판정 2 줄 중 **2 줄 달성**" in text
    assert "전제" in text
    # 전제 두 줄(완주율·선생님 대비 점수)의 이름이 "전제" 절 아래 있다 — 헤드라인 숫자에는
    # 안 들어가지만 성적표에서 사라지진 않는다.
    assert "완주율" in text.split("전제")[1]
    assert "선생님 대비 점수" in text.split("전제")[1]


def test_항목별_표의_중대_합계는_완주_판만_센다():
    """`completed_only` 를 빼면 조기 종료(timeout)한 판이 도달 못 한 구간의 위반을 시트에 못

    남겨 중대가 실제보다 적게(또는, 이 예시처럼 완주 못한 판에 중대가 더 있으면 많게) 잡힌다.
    `major_total(violation_counts(completed_only(ev)))` 순서를 지켜야 한다(Task 6 정본).
    """
    module = _load_report_m4a_module()
    from vtd_rl.eval.verdict import completed_only, major_total

    # 완주 판 1개(중대 1건) + 미완주 판 1개(도달한 구간에 중대 2건 더) — completed_only 를
    # 빼면 합계가 1이 아니라 3이 된다.
    episodes = [
        EpisodeOutcome("A", 0, "goal", 10, 0.0, 90.0, [{2: "major"}]),
        EpisodeOutcome("A", 1, "timeout", 5, 0.0, 0.0, [{2: "major"}, {2: "major"}]),
    ]
    ev = {"episodes": episodes}
    assert major_total(violation_counts(completed_only(ev))) == 1   # 완주 판만 셌을 때의 참값

    m3_ev = {module.STAGE1_LABEL: ev, module.STAGE2_LABEL: ev}
    m4a_ev = {module.STAGE1_LABEL: ev, module.STAGE2_LABEL: ev}
    teacher_ev = {module.STAGE1_LABEL: ev, module.STAGE2_LABEL: ev}
    text = "\n".join(module._violation_table_lines([module.STAGE1_LABEL, module.STAGE2_LABEL],
                                                    m3_ev, m4a_ev, teacher_ev))
    assert "중대 합계" in text
    assert "| **중대 합계(완주 판 기준)** | — | 1 | — | 1 | — | 1 |" in text
    assert "| **중대 합계(완주 판 기준)** | — | 3 | — | 3 | — | 3 |" not in text


@pytest.mark.slow
def test_성적표가_중대_합계와_시드_편차를_담는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    root = str(tmp_path / "sweep")
    sw = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                         os.path.join(REPO, "scripts", "sweep_ppo.py"),
                         "--out-root", root, "--name", "smoke", "--seeds", "0", "1",
                         "--", "--smoke"],
                        capture_output=True, text=True, env=env, cwd=REPO, timeout=3600)
    assert sw.returncode == 0, sw.stderr[-3000:]

    report = str(tmp_path / "m4b.md")
    made = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                           os.path.join(REPO, "scripts", "report_m4a.py"),
                           "--run", os.path.join(root, "smoke-s0"),
                           "--sweep", os.path.join(root, "sweep.json"),
                           "--out", report, "--skip-eval"],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert made.returncode == 0, made.stderr[-3000:]
    text = open(report, encoding="utf-8").read()
    for needed in ("중대 합계", "시드 편차", "rollout_return_mean", "drift_rel"):
        assert needed in text, needed
