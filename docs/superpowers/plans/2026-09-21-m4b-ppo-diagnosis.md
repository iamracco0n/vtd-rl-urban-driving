# M4b 구현 계획 — 목표를 바로 세우고, 붕괴를 보이게 만든다

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M4a 가 남긴 세 가지를 처리한다 — **틀린 목표**(항목 두 개만 세다가 풍선 누르기를 통과시켰다), **보이지 않는 학습 신호**(PPO 가 최대화하는 양을 한 번도 로깅하지 않았다), **깨지지 않는 모방 앵커**(σ 를 하한에 붙박아 탐색을 없앴다). 커리큘럼 단계 ③④⑤ 확장은 이 뒤로 미룬다.

**Architecture:** 새 모듈은 둘뿐이다. `vtd_rl/eval/verdict.py` 는 목표 판정을 한 곳에 모아 학습 스크립트와 성적표가 **같은 규칙**을 쓰게 한다(M4a 에서는 성적표에만 있었다). `vtd_rl/rl/diagnostics.py` 는 롤아웃 리턴·종료 사유·초기 정책으로부터의 드리프트를 재서 `log.jsonl` 에 남긴다. 나머지는 기존 파일의 작은 수정이다 — 모방 손실에 σ 분리 모드를 더하고, 시드 스윕 러너를 붙인다.

**Tech Stack:** Python 3.10, PyTorch 2.14, Gymnasium 1.3, numpy 1.26.4, pytest. 개발·테스트는 lab-main(16 코어), 본 학습은 OMEN(`user-OMEN` 100.87.135.28, 32 코어).

**Spec:** `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md` (§5 보상, §6.3 PPO, §6.4 진급, §6.5 두 머신)

**직전 성적표(반드시 읽을 것):** `docs/reports/m4a-ppo.md` 와 해석 `docs/reports/m4a-ppo-notes.md`

## Global Constraints

- 커밋 메시지에 Claude 표기(`Co-Authored-By`, `Claude-Session`, "Generated with")를 **넣지 않는다**.
- 규칙 스택 서브모듈(`third_party/rule_stack`)은 **수정하지 않는다**.
- 공개 레포에 주행 CSV, `runs/`, 체크포인트(`*.pt`), 데이터 조각(`*.npz`), 주최측 자료(`*.xodr`, `*.xml`)를 커밋하지 않는다. **성적표 md 만 커밋한다.**
- **numpy 1.26.4 고정, torch >=2.4,<3.**
- 판정은 **종료 코드 0**. `"N passed"` 를 grep 하지 마라 — `"1 failed"` 를 삼킨다.
- **심판·채점기·세계·환경(`vtd_rl/env/`, `vtd_rl/referee/`, `vtd_rl/world/`)의 문턱과 규칙은 바꾸지 않는다.** 보상 계수(`RewardConfig`)도 이 계획에서는 건드리지 않는다.
- 문서·주석·커밋 메시지는 **한국어**로 쓴다(기존 코드가 전부 한국어 주석이다).
- **성적표 숫자는 전부 스크립트가 생성한다.** 손으로 옮겨 적지 마라(M3 에서 그 때문에 세 번 고쳤다).
- **기준을 낮추지 않는다.** 테스트가 통과하게 하려고 단언을 약하게 고치지 마라.

## M4a 가 남긴 사실 (이 계획의 출발점)

실행 3 회, 전부 OMEN 30 환경·3M 스텝·시드 0·`--device cpu`·M3 학생(`runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt`)에서 출발.

| 단계② 지표 | M3 학생 | run1 기본 | run2 엔트로피 0.05 | run3 +모방반감기 200k | 선생님 |
|---|---:|---:|---:|---:|---:|
| ⑦ 적색 정지 중대 | 64 | 68 | **19** | 55 | 1 |
| ② 보호구역 중대 | 15 | 15 | 11 | 21 | 0 |
| ③ 차로 유지 중대 | 0 | 0 | **18** | — | 0 |
| 완주율 | 100% | 100% | 94.4% | 100% | 100% |
| **점수** | **94.2** | **94.2** | **90.7** | 92.2 | 99.5 |

- **run2 가 항목 ⑦ 을 70% 줄였지만 항목 ③ 을 0 → 18 로 터뜨렸고 총점은 내려갔다.** M4a 목표가 ⑦·② 만 세서 이 퇴행이 판정에 안 잡혔다.
- **단계 ① 은 더 나쁘다.** run2 의 단계① 중대 합계가 항목② 15 + 항목③ 18 + 항목④(중앙선 침범) 3 = **36** 으로, M3 학생의 **15** 에서 두 배 넘게 늘었다. 단계② 합계는 48 로 M3(79)보다 줄었다 — 즉 **한 단계에서 줄인 것을 다른 단계에서 되돌려 받았다.** 이것이 목표 4 번을 넣는 이유다.
- 세 실행 모두 1M~1.5M 이후 완주율이 무너진다. run1 은 3M 에서 단계② 0%.
- `log_std[0]`(조향)은 run1·run2 에서 3M 내내 하한 −2.000. run3 은 모방 계수가 사라지자 −1.250 까지 올랐지만 **완주율이 0% 로 고착**했다.
- 모방 손실이 `log_std[0]` 에 주는 기울기 **+0.9392**, 기본 엔트로피 계수의 반대 힘 0.005 = **0.5%**(188:1). 학생 조향 잔차 RMS 0.0334(≈1.2°).
- **PPO 가 최대화하는 확률적 롤아웃 리턴은 한 번도 로깅되지 않았다.** 인용된 보상은 전부 결정적 평가 보상(프록시)이다.

## 목표(성공 기준)

M4a 의 목표 네 줄 중 **항목별 두 줄을 버린다.** 스펙 §6.4 는 원래 "완주율 ≥ 90% 이고 `score_fma` 점수가 선생님보다 크게 낮지 않으면" 이라고 **총점**으로 쓰여 있었다 — 항목별 문턱은 M4a 계획이 얹은 것이고, 그게 풍선 누르기를 통과시켰다. 스펙으로 되돌리고, M4a 가 **묻지 않았던 것**을 더한다.

| # | 목표 | 기준 | 근거 |
|---|---|---|---|
| 1 | 완주율 | 단계①② 모두 **≥ 90%** | 스펙 §6.4 |
| 2 | 선생님 대비 점수 | 단계①② 모두 **≥ 선생님 − 10**(완주 못 한 판 0점) | 스펙 §6.4 |
| 3 | **출발점 대비 점수** | 단계①② 모두 **> M3 학생**(단계① 98.5, 단계② 94.2) | **M4a 가 안 물었다.** PPO 가 출발점을 못 넘으면 할 이유가 없다 |
| 4 | **전 항목 중대 위반 합계** | 단계① **≤ 15**, 단계② **≤ 79**(= M3 학생 수준, 늘지 말 것) | 풍선 누르기 방지 — run2 가 ⑦ 을 줄이며 ③ 을 터뜨린 것을 잡는다 |

항목별 건수는 **진단 표로만 싣는다**(문턱이 아니다). 네 줄을 모두 채우면 달성이다. 못 채우면 **기준을 고치지 말고** 숫자와 관찰을 성적표에 적는다.

이 목표 자체도 완벽하지 않다 — 3·4 번은 M3 학생이라는 한 점에 묶여 있고, M3 자체가 선생님보다 한참 아래다. 그 한계를 성적표에 적는다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `vtd_rl/eval/__init__.py`, `vtd_rl/eval/verdict.py` | 목표 판정 한 곳 — 문턱 상수, `GoalVerdict`, `judge()`. 학습 스크립트와 성적표가 공유 |
| `vtd_rl/rl/diagnostics.py` | 롤아웃 리턴·종료 사유·정책 드리프트 계측 |
| `vtd_rl/policy/train.py` (수정) | `policy_loss` 에 σ 분리 모드 |
| `vtd_rl/rl/ppo.py` (수정) | `PPOConfig.imitation_sigma`, `imitation_loss` 가 그 모드를 넘김 |
| `scripts/train_ppo.py` (수정) | 계측 연결, `--imitation-sigma`, 판정 요약 |
| `scripts/sweep_ppo.py` | 시드 스윕 러너 + 편차 집계 |
| `scripts/report_m4a.py` (수정) | 목표 판정을 `verdict.py` 로 위임, 진단 표, 시드 편차 |
| `docs/reports/m4b-ppo.md` | 완료 증거(스크립트가 생성) |

---

### Task 1: 목표 판정을 한 곳에 모으고 총 감점 기준으로 다시 건다

**Files:**
- Create: `vtd_rl/eval/__init__.py`(빈 파일), `vtd_rl/eval/verdict.py`, `tests/eval/__init__.py`(빈 파일), `tests/eval/test_verdict.py`
- Modify: `docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md`(§6.4 에 한 줄)

**Interfaces:**
- Produces:
  - `GOAL_RATE_MIN = 0.9`, `TEACHER_SCORE_SLACK = 10.0`
  - `M3_SCORE = {"stage1": 98.5, "stage2": 94.2}` — 2026-09-21 시드 3 개 실측
  - `M3_MAJOR_TOTAL = {"stage1": 15, "stage2": 79}` — 같은 실측(단계① 항목② 15; 단계② 항목② 15 + 항목⑦ 64)
  - `@dataclass(frozen=True) GoalVerdict: name: str; ok: bool; line: str`
  - `major_total(counts: dict) -> int` — `violation_counts()` 결과에서 중대만 합산
  - `judge(student: dict, teacher: dict, major_totals: dict, eval_seeds: int) -> list[GoalVerdict]`
    - `student`·`teacher` 는 `{"stage1": <evaluate_policy 결과>, "stage2": ...}`
    - `major_totals` 는 `{"stage1": int, "stage2": int}`
    - 네 줄을 순서대로 돌려준다. `eval_seeds != 3` 이면 3·4 번은 `ok=False` 가 아니라 **보류**로 표시하고 `line` 에 이유를 적는다(M3 기준값이 시드 3 개 합계라 시드 수가 다르면 비교가 성립하지 않는다)
- Consumes: 없음(순수 함수 — 평가 결과 dict 만 받는다)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/__init__.py` — 빈 파일.

`tests/eval/test_verdict.py`
```python
import pytest

from vtd_rl.eval.verdict import (GOAL_RATE_MIN, M3_MAJOR_TOTAL, M3_SCORE,
                                 TEACHER_SCORE_SLACK, judge, major_total)


def ev(goal_rate, mean_score):
    return {"goal_rate": goal_rate, "mean_score": mean_score}


def both(goal_rate, mean_score):
    return {"stage1": ev(goal_rate, mean_score), "stage2": ev(goal_rate, mean_score)}


TEACHER = {"stage1": ev(1.0, 99.6), "stage2": ev(1.0, 99.5)}


def test_중대만_합산한다():
    counts = {2: {"minor": 3, "major": 15}, 7: {"minor": 0, "major": 64}, 3: {"minor": 12, "major": 0}}
    assert major_total(counts) == 79


def test_빈_집계는_0():
    assert major_total({}) == 0


def test_네_줄을_순서대로_준다():
    v = judge(both(1.0, 99.0), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert [x.name for x in v] == ["완주율", "선생님 대비 점수", "출발점 대비 점수", "전 항목 중대 위반"]


def test_전부_통과하는_경우():
    v = judge(both(1.0, 99.0), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert all(x.ok for x in v)


def test_완주율이_모자라면_첫_줄만_깨진다():
    s = {"stage1": ev(0.89, 99.0), "stage2": ev(1.0, 99.0)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[0].ok is False and v[1].ok and v[2].ok and v[3].ok


def test_선생님보다_10점_넘게_낮으면_두번째가_깨진다():
    s = {"stage1": ev(1.0, 89.5), "stage2": ev(1.0, 99.0)}      # 99.6 - 10 = 89.6
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[1].ok is False


def test_M3_와_같기만_하면_세번째가_깨진다():
    # 목표는 "M3 보다 나을 것" 이다 — 같으면 개선이 아니다
    s = {"stage1": ev(1.0, M3_SCORE["stage1"]), "stage2": ev(1.0, M3_SCORE["stage2"])}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[2].ok is False


def test_중대_합계가_M3_보다_늘면_네번째가_깨진다():
    over = {"stage1": M3_MAJOR_TOTAL["stage1"], "stage2": M3_MAJOR_TOTAL["stage2"] + 1}
    v = judge(both(1.0, 99.0), TEACHER, over, eval_seeds=3)
    assert v[3].ok is False


def test_M4a_run2_는_새_목표에서_세번째_네번째_모두_걸린다():
    """M4a 가 "달성" 으로 판정했던 실행이 새 기준에서 왜 걸리는지 고정한다.

    2026-09-21 실측(`docs/reports/m4a-ppo.md`, 시드 3개):
      단계① 중대 = 항목② 15 + 항목③ 18 + 항목④ 3 = **36** (M3 는 15) → 늘었다
      단계② 중대 = 항목② 11 + 항목③ 18 + 항목⑦ 19 = **48** (M3 는 79) → 줄었다
    항목 ⑦ 만 보면 64 → 19 로 좋아졌지만, 그 대가로 차로 유지·중앙선이 터져
    단계① 합계가 두 배 넘게 늘었고 점수도 M3 아래다. M4a 목표는 이걸 못 잡았다.
    """
    s = {"stage1": ev(1.0, 96.9), "stage2": ev(0.944, 90.7)}
    v = judge(s, TEACHER, {"stage1": 36, "stage2": 48}, eval_seeds=3)
    assert v[0].ok and v[1].ok                 # 완주율·선생님 대비 점수는 통과한다
    assert v[2].ok is False and "94.2" in v[2].line     # 단계② 90.7 < M3 94.2
    assert v[3].ok is False and "36" in v[3].line       # 단계① 36 > M3 15


def test_시드가_3이_아니면_3_4번은_보류():
    v = judge(both(1.0, 99.0), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=1)
    assert v[0].ok and v[1].ok
    assert v[2].ok is False and "보류" in v[2].line
    assert v[3].ok is False and "보류" in v[3].line


def test_문턱_상수가_스펙_값이다():
    assert GOAL_RATE_MIN == 0.9 and TEACHER_SCORE_SLACK == 10.0
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/eval/test_verdict.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.eval'`

- [ ] **Step 3: 구현**

`vtd_rl/eval/__init__.py` — 빈 파일.

`vtd_rl/eval/verdict.py`
```python
"""목표 판정 — 학습 스크립트와 성적표가 **같은 규칙**을 쓰게 한 곳에 모은다.

M4a 는 판정이 성적표에만 있었고, 목표를 **항목 두 개**(⑦ 적색 정지·② 보호구역)의 절대 건수로
걸었다. 그 결과 run2 가 항목 ⑦ 을 64 → 19 로 줄이면서 항목 ③(차로 유지) 중대를 0 → 18 로
터뜨렸는데도 판정은 "달성" 이 나왔다 — 총점은 94.2 → 90.7 로 내려갔는데도. 풍선의 한쪽을
눌러 다른 쪽을 부풀리는 정책이 통과한 것이다.

그래서 스펙 §6.4 의 원래 문구(완주율과 **총점**)로 되돌리고, M4a 가 묻지 않았던 두 가지를
더한다 — **출발점(M3 학생)보다 나은가**, 그리고 **중대 위반 총합이 늘지 않았는가**.
항목별 건수는 진단으로만 본다.
"""
from dataclasses import dataclass

GOAL_RATE_MIN = 0.9
TEACHER_SCORE_SLACK = 10.0

# 2026-09-21 실측(시드 3개, `runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt`).
# 단계① 중대: 항목② 15. 단계② 중대: 항목② 15 + 항목⑦ 64 = 79.
M3_SCORE = {"stage1": 98.5, "stage2": 94.2}
M3_MAJOR_TOTAL = {"stage1": 15, "stage2": 79}

# M3 기준값이 시드 3개 합계라, 시드 수가 다르면 3·4번은 비교 자체가 성립하지 않는다.
REQUIRED_EVAL_SEEDS = 3


@dataclass(frozen=True)
class GoalVerdict:
    name: str
    ok: bool
    line: str


def major_total(counts: dict) -> int:
    """`violation_counts()` 결과에서 **중대만** 합산한다 — 경미는 감점이 작아 총점에 묻힌다."""
    return sum(v.get("major", 0) for v in counts.values())


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def judge(student: dict, teacher: dict, major_totals: dict, eval_seeds: int) -> list:
    s1, s2 = student["stage1"], student["stage2"]
    t1, t2 = teacher["stage1"], teacher["stage2"]
    held = eval_seeds != REQUIRED_EVAL_SEEDS
    hold = (f" — **보류**(M3 기준값이 시드 {REQUIRED_EVAL_SEEDS}개 합계인데 이번 평가는"
            f" 시드 {eval_seeds}개다. 같은 시드 수로 다시 재야 비교가 성립한다)")

    goal_ok = s1["goal_rate"] >= GOAL_RATE_MIN and s2["goal_rate"] >= GOAL_RATE_MIN
    score_ok = (s1["mean_score"] >= t1["mean_score"] - TEACHER_SCORE_SLACK
                and s2["mean_score"] >= t2["mean_score"] - TEACHER_SCORE_SLACK)
    beats_m3 = (s1["mean_score"] > M3_SCORE["stage1"] and s2["mean_score"] > M3_SCORE["stage2"])
    majors_ok = (major_totals["stage1"] <= M3_MAJOR_TOTAL["stage1"]
                 and major_totals["stage2"] <= M3_MAJOR_TOTAL["stage2"])

    return [
        GoalVerdict("완주율", goal_ok,
                    f"완주율 단계①② 모두 ≥{_pct(GOAL_RATE_MIN)} → **{'달성' if goal_ok else '미달'}**"
                    f" (실측: 단계① {_pct(s1['goal_rate'])}, 단계② {_pct(s2['goal_rate'])})"),
        GoalVerdict("선생님 대비 점수", score_ok,
                    f"점수가 선생님보다 {TEACHER_SCORE_SLACK:.0f}점 넘게 낮지 않을 것 →"
                    f" **{'달성' if score_ok else '미달'}** (실측: 단계① {s1['mean_score']:.1f} vs"
                    f" 선생님 {t1['mean_score']:.1f}, 단계② {s2['mean_score']:.1f} vs"
                    f" 선생님 {t2['mean_score']:.1f})"),
        GoalVerdict("출발점 대비 점수", beats_m3 and not held,
                    f"점수가 출발점(M3 학생)보다 나을 것 →"
                    f" **{'달성' if beats_m3 and not held else '미달'}** (실측: 단계①"
                    f" {s1['mean_score']:.1f} vs M3 {M3_SCORE['stage1']:.1f}, 단계②"
                    f" {s2['mean_score']:.1f} vs M3 {M3_SCORE['stage2']:.1f})"
                    + (hold if held else "")),
        GoalVerdict("전 항목 중대 위반", majors_ok and not held,
                    f"중대 위반 **합계**가 M3 학생보다 늘지 않을 것 →"
                    f" **{'달성' if majors_ok and not held else '미달'}** (실측: 단계①"
                    f" {major_totals['stage1']} vs M3 {M3_MAJOR_TOTAL['stage1']}, 단계②"
                    f" {major_totals['stage2']} vs M3 {M3_MAJOR_TOTAL['stage2']})"
                    + (hold if held else "")),
    ]
```

스펙 §6.4 에 한 줄 더한다(기존 줄 아래):
```markdown
- 항목별 위반 건수는 **진단으로만 본다.** 목표로 걸면 한 항목을 줄이며 다른 항목을 터뜨리는 정책이 통과한다(M4a 실측: 항목⑦ 64→19 를 얻으며 항목③ 0→18, 총점 94.2→90.7). 규칙 준수는 **총점**과 **중대 위반 합계**로 건다.
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/eval tests/eval docs/superpowers/specs
git commit -m "M4b — 목표 판정을 한 곳에 모으고 항목별 문턱 대신 총점·중대 합계로 건다"
```

---

### Task 2: 계측 — 롤아웃 리턴·종료 사유·정책 드리프트

**Files:**
- Create: `vtd_rl/rl/diagnostics.py`, `tests/rl/test_diagnostics.py`
- Test: `tests/rl/test_diagnostics.py`

**Interfaces:**
- Consumes: `torch`, `numpy`
- Produces:
  - `class ReturnTracker(n_envs)` — 롤아웃을 돌며 환경별 누적 보상을 이어 붙이고, 판이 끝나면 그 리턴을 담는다
    - `.add(reward, done)` — `reward`·`done` 은 길이 `n_envs` 인 numpy 배열(`done` = `terminated | truncated`)
    - `.stats() -> dict` — `{"rollout_return_mean": float|None, "rollout_return_n": int, "rollout_len_mean": float|None}`. 끝난 판이 하나도 없으면 평균은 `None`
    - `.reset()` — 담아 둔 리턴만 비운다(진행 중인 누적은 롤아웃 경계를 넘어 이어진다)
  - `class OutcomeCounter()` — `.add(infos)` 로 벡터 환경 `infos` 에서 `outcome` 을 세고, `.stats() -> dict` 가 `{"outcome_goal": n, "outcome_collision": n, ...}` 를 준다. 본 적 없는 사유도 키가 생긴다
  - `policy_drift(net, ref_state: dict) -> dict` — 현재 정책 파라미터와 기준 상태의 거리. `{"drift_l2": float, "drift_rel": float}`. `drift_rel` 은 기준 노름으로 나눈 값
  - `snapshot_policy(net) -> dict` — `net.policy` 의 파라미터를 detach·clone 해 담는다(기준 상태 만들기)
- **왜 필요한가:** M4a 는 PPO 가 최대화하는 **확률적 롤아웃 리턴**을 한 번도 로깅하지 않아, "학습은 되는데 평가가 나쁘다" 와 "학습 자체가 안 된다" 를 가를 수 없었다. 종료 사유와 드리프트는 1M 이후 붕괴가 **무엇으로** 나타나는지(충돌인가 이탈인가 정체인가, 정책이 얼마나 멀리 갔는가)를 보이게 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_diagnostics.py`
```python
import numpy as np
import torch

from vtd_rl.rl.diagnostics import (OutcomeCounter, ReturnTracker, policy_drift,
                                   snapshot_policy)


def test_끝난_판의_리턴만_담는다():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 2.0]), np.array([False, False]))
    t.add(np.array([3.0, 4.0]), np.array([True, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 1
    assert abs(s["rollout_return_mean"] - 4.0) < 1e-9      # 0번 환경: 1+3
    assert abs(s["rollout_len_mean"] - 2.0) < 1e-9


def test_끝난_판이_없으면_평균은_None():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 1.0]), np.array([False, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 0 and s["rollout_return_mean"] is None
    assert s["rollout_len_mean"] is None


def test_판이_끝나면_누적이_0에서_다시_시작한다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([2.0]), np.array([True]))
    s = t.stats()
    assert s["rollout_return_n"] == 2
    assert abs(s["rollout_return_mean"] - 3.5) < 1e-9      # (5 + 2) / 2


def test_reset_은_담은_것만_비우고_진행중_누적은_남긴다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([1.0]), np.array([False]))              # 진행 중 누적 1.0
    t.reset()
    assert t.stats()["rollout_return_n"] == 0
    t.add(np.array([2.0]), np.array([True]))
    assert abs(t.stats()["rollout_return_mean"] - 3.0) < 1e-9   # 1 + 2 — 경계를 넘어 이어졌다


def test_종료_사유를_센다():
    c = OutcomeCounter()
    c.add({"outcome": np.array(["goal", "collision", "running"], dtype=object)})
    c.add({"outcome": np.array(["goal", "stalled", "running"], dtype=object)})
    s = c.stats()
    assert s["outcome_goal"] == 2 and s["outcome_collision"] == 1 and s["outcome_stalled"] == 1
    assert "outcome_running" not in s                      # 진행 중은 종료 사유가 아니다


def test_outcome_키가_없으면_아무것도_안_센다():
    c = OutcomeCounter()
    c.add({})
    assert c.stats() == {}


def test_드리프트는_움직인_만큼_커진다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    assert policy_drift(net, ref)["drift_l2"] == 0.0
    with torch.no_grad():
        net.policy.log_std.add_(1.0)
    d = policy_drift(net, ref)
    assert abs(d["drift_l2"] - 2.0 ** 0.5) < 1e-5          # 두 차원에 각각 1.0
    assert d["drift_rel"] > 0.0


def test_스냅샷은_이후_변경에_영향받지_않는다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    with torch.no_grad():
        net.policy.log_std.add_(3.0)
    assert policy_drift(net, ref)["drift_l2"] > 1.0        # clone 이 아니면 0 이 나온다
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_diagnostics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vtd_rl.rl.diagnostics'`

- [ ] **Step 3: 구현**

`vtd_rl/rl/diagnostics.py`
```python
"""학습이 무엇을 하고 있는지 보이게 하는 계측.

M4a 는 세 실행이 1M~1.5M 이후 무너졌는데 **왜** 무너졌는지 못 밝혔다. 이유는 계측이 없어서다:
PPO 가 실제로 최대화하는 **확률적 롤아웃 리턴**이 한 번도 로깅되지 않아, 인용할 수 있는
보상은 결정적 평가 보상(프록시)뿐이었다. 종료 사유와 정책 드리프트도 마찬가지로 없었다.

세 가지를 잰다 — 롤아웃 리턴(PPO 의 목적함수), 종료 사유(붕괴가 충돌·이탈·정체 중 무엇으로
나타나는가), 초기 정책으로부터의 거리(정책이 얼마나 멀리 갔는가).
"""
import numpy as np
import torch

# 판이 아직 안 끝난 상태 — 종료 사유가 아니므로 세지 않는다.
RUNNING = "running"


class ReturnTracker:
    """환경별로 보상을 누적하다가 판이 끝나면 그 리턴을 담는다.

    `reset()` 은 **담아 둔 리턴만** 비운다. 진행 중인 누적은 롤아웃 경계를 넘어 이어져야 한다 —
    판 하나가 롤아웃 여러 개에 걸치기 때문이다(판은 2500~5000 걸음, 롤아웃은 256 걸음).
    """

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self._running = np.zeros(n_envs, dtype=np.float64)
        self._steps = np.zeros(n_envs, dtype=np.int64)
        self.reset()

    def reset(self):
        self._returns: list = []
        self._lengths: list = []

    def add(self, reward, done):
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        self._running += reward
        self._steps += 1
        for i in np.nonzero(done)[0]:
            self._returns.append(float(self._running[i]))
            self._lengths.append(int(self._steps[i]))
            self._running[i] = 0.0
            self._steps[i] = 0

    def stats(self) -> dict:
        n = len(self._returns)
        return {"rollout_return_mean": (sum(self._returns) / n) if n else None,
                "rollout_return_n": n,
                "rollout_len_mean": (sum(self._lengths) / n) if n else None}


class OutcomeCounter:
    """벡터 환경 `infos` 에서 종료 사유를 센다 — 붕괴가 무엇으로 나타나는지 보려는 것."""

    def __init__(self):
        self._counts: dict = {}

    def add(self, infos: dict):
        values = infos.get("outcome")
        if values is None:
            return
        for v in np.asarray(values, dtype=object).reshape(-1):
            if v is None or v == RUNNING:
                continue
            key = f"outcome_{v}"
            self._counts[key] = self._counts.get(key, 0) + 1

    def stats(self) -> dict:
        return dict(self._counts)


def snapshot_policy(net) -> dict:
    """정책 파라미터를 떼어 복사한다 — clone 을 빼면 이후 갱신이 기준까지 따라 움직인다."""
    return {k: v.detach().clone() for k, v in net.policy.state_dict().items()}


def policy_drift(net, ref_state: dict) -> dict:
    """현재 정책이 기준에서 얼마나 멀어졌나. `approx_kl` 은 갱신 한 걸음의 크기만 보지만,

    이 값은 **누적된 이동**을 본다 — M4a 의 붕괴는 걸음마다는 작고(KL 0.011~0.016) 누적으로만
    큰 형태였다.
    """
    with torch.no_grad():
        sq, ref_sq = 0.0, 0.0
        for k, v in net.policy.state_dict().items():
            r = ref_state[k].to(v.device)
            sq += float((v - r).pow(2).sum())
            ref_sq += float(r.pow(2).sum())
    l2 = sq ** 0.5
    return {"drift_l2": l2, "drift_rel": l2 / (ref_sq ** 0.5) if ref_sq > 0 else 0.0}
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/rl/diagnostics.py tests/rl/test_diagnostics.py
git commit -m "M4b 계측 — 롤아웃 리턴·종료 사유·정책 드리프트"
```

---

### Task 3: 모방 손실에서 σ 를 분리한다 (평균은 그대로 배운다)

**Files:**
- Modify: `vtd_rl/policy/train.py`, `vtd_rl/rl/ppo.py`
- Test: `tests/policy/test_train.py`(추가), `tests/rl/test_ppo.py`(추가)

**Interfaces:**
- Produces:
  - `TrainConfig.sigma_grad: bool = True` — `False` 면 NLL 을 **`log_std.detach()`** 로 계산한다. 평균과 지시등에는 기울기가 그대로 흐르고 `log_std` 에만 안 흐른다
  - `PPOConfig.imitation_sigma: str = "learn"` — `"learn"` 또는 `"detach"`. `imitation_loss` 가 이 값에 따라 `TrainConfig(sigma_grad=...)` 를 골라 넘긴다
- Consumes: `policy_loss`, `_IMITATION_TRAIN_CFG`
- **왜 이렇게 하나:** M4a 실측으로 모방 손실이 `log_std[0]` 을 −3.40 까지 끌어내리려 하고(기울기 +0.9392) 하한 −2.0 이 겨우 붙들고 있다. 기본 엔트로피 계수의 반대 힘은 그 **0.5%** 다. 그런데 모방 계수를 통째로 빨리 줄인 run3 은 σ 가 풀리는 대신 **완주율이 0% 로 붕괴**했다 — 평균까지 함께 풀렸기 때문이다. 그래서 **평균은 선생님에 묶어 두고 σ 만 푼다.** 이것이 M4a 성적표의 M4b 권고 2 번이고, 아직 시험된 적이 없다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/policy/test_train.py` 에 더한다(기존 테스트는 그대로 둔다). **그 파일의 임포트 줄은 지금 `from vtd_rl.policy.train import TrainConfig, evaluate_labels, train_epochs` 라 `policy_loss` 가 없다 — 더해라.**
```python
def test_시그마를_분리하면_log_std에_기울기가_안_간다():
    torch.manual_seed(0)
    net = DrivePolicy(PolicyConfig(trunk=(32, 32)))
    batch = next(iter(toy_dataset(64).batches(32, generator=torch.Generator().manual_seed(0))))

    net.zero_grad()
    policy_loss(net, batch, TrainConfig(sigma_grad=True))[0].backward()
    assert net.log_std.grad is not None and torch.any(net.log_std.grad != 0.0)
    learn_mean_grad = net.mean.weight.grad.clone()

    net.zero_grad()
    policy_loss(net, batch, TrainConfig(sigma_grad=False))[0].backward()
    assert net.log_std.grad is None or torch.all(net.log_std.grad == 0.0)
    # 평균 쪽 기울기는 살아 있어야 한다 — σ 만 떼는 것이지 모방을 끄는 게 아니다
    assert torch.any(net.mean.weight.grad != 0.0)
    # 그리고 σ 를 상수로 본 만큼 평균 기울기의 방향이 달라지지는 않는다(같은 var 로 나눈다)
    assert torch.allclose(net.mean.weight.grad, learn_mean_grad, atol=1e-6)


def test_시그마_분리_기본값은_학습이다():
    assert TrainConfig().sigma_grad is True
```

`tests/rl/test_ppo.py` 에 더한다:
```python
def test_모방_시그마_모드가_실제로_전달된다(small_ac):
    net = small_ac()
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))

    net.zero_grad()
    imitation_loss(net, batch, PPOConfig(imitation_sigma="learn"))[0].backward()
    assert torch.any(net.policy.log_std.grad != 0.0)

    net.zero_grad()
    imitation_loss(net, batch, PPOConfig(imitation_sigma="detach"))[0].backward()
    assert net.policy.log_std.grad is None or torch.all(net.policy.log_std.grad == 0.0)


def test_모방_시그마_기본값은_learn():
    assert PPOConfig().imitation_sigma == "learn"


def test_모방_시그마에_이상한_값을_주면_거부한다(small_ac):
    net = small_ac()
    batch = next(iter(toy_dagger(64).batches(32, generator=torch.Generator().manual_seed(0))))
    with pytest.raises(ValueError):
        imitation_loss(net, batch, PPOConfig(imitation_sigma="아무거나"))
```

`tests/rl/test_ppo.py` 상단 임포트에 `import pytest` 가 없으면 더한다.

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/policy/test_train.py tests/rl/test_ppo.py -q`
Expected: FAIL — `TypeError: TrainConfig.__init__() got an unexpected keyword argument 'sigma_grad'`

- [ ] **Step 3: 구현**

`vtd_rl/policy/train.py`:
- `TrainConfig` 에 필드를 더한다.
```python
    sigma_grad: bool = True         # False 면 NLL 을 log_std.detach() 로 계산한다(M4b: σ 만 푼다)
```
- `policy_loss` 의 `log_std` 사용을 고친다:
```python
def policy_loss(net, batch, cfg: TrainConfig):
    vec, objs, mask, control, turn = batch
    mean, log_std, logits = net(vec, objs, mask)
    # σ 를 떼면 평균·지시등에는 기울기가 그대로 흐르고 log_std 에만 안 흐른다. M4a 실측에서
    # 모방 손실이 log_std[0] 을 하한에 붙박아(기울기 +0.9392) 조향 탐색을 없앴는데, 모방을
    # 통째로 줄이면(run3) 평균까지 풀려 완주율이 0% 로 무너졌다 — 그래서 σ 만 뗀다.
    nll_log_std = log_std.detach() if not cfg.sigma_grad else log_std
    var = (2.0 * nll_log_std).exp()
    nll = 0.5 * (((control - mean) ** 2) / var + 2.0 * nll_log_std + LOG_2PI)
    control_loss = nll.sum(dim=-1).mean()
    turn_loss = nn.functional.cross_entropy(logits, turn)
    total = control_loss + cfg.turn_weight * turn_loss
    return total, {"control": float(control_loss.item()), "turn": float(turn_loss.item()),
                   "total": float(total.item())}
```

`vtd_rl/rl/ppo.py`:
- `PPOConfig` 에 필드를 더한다.
```python
    imitation_sigma: str = "learn"   # "learn" | "detach" — detach 면 모방이 σ 를 안 건드린다
```
- 모듈 상수 옆에 짝을 하나 더 만든다(기존 `_IMITATION_TRAIN_CFG` 는 그대로 둔다).
```python
_IMITATION_TRAIN_CFG_DETACH = dataclasses.replace(_IMITATION_TRAIN_CFG, sigma_grad=False)
_IMITATION_CFGS = {"learn": _IMITATION_TRAIN_CFG, "detach": _IMITATION_TRAIN_CFG_DETACH}
```
(`import dataclasses` 가 없으면 더한다.)
- `imitation_loss` 를 고친다:
```python
def imitation_loss(net, dagger_batch, cfg: PPOConfig):
    """M3 라벨에 대한 로그가능도 — `vtd_rl.policy.train.policy_loss` 를 그대로 쓴다.

    `net` 은 `ActorCritic`(가치 머리 포함)이지만, `policy_loss` 는 `DrivePolicy` 를
    직접 호출하므로(`net(vec, objs, mask) -> mean, log_std, logits`) 반드시 `net.policy` 를
    넘긴다. `cfg.imitation_sigma` 가 `"detach"` 면 σ 에는 기울기를 안 보낸다.
    """
    try:
        train_cfg = _IMITATION_CFGS[cfg.imitation_sigma]
    except KeyError:
        raise ValueError(f"imitation_sigma 는 {sorted(_IMITATION_CFGS)} 중 하나여야 한다:"
                         f" {cfg.imitation_sigma!r}")
    return policy_loss(net.policy, dagger_batch, train_cfg)
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0` — 기존 DAgger 경로는 `sigma_grad` 기본값 `True` 라 동작이 안 바뀐다

- [ ] **Step 5: 커밋**

```bash
git add vtd_rl/policy/train.py vtd_rl/rl/ppo.py tests/policy/test_train.py tests/rl/test_ppo.py
git commit -m "M4b — 모방 손실에서 σ 만 분리하는 모드(평균은 선생님에 묶어 둔다)"
```

---

### Task 4: 학습 스크립트에 계측·σ 모드·판정을 연결한다

**Files:**
- Modify: `scripts/train_ppo.py`
- Test: `tests/rl/test_train_ppo.py`(추가)

**Interfaces:**
- Consumes: `ReturnTracker`, `OutcomeCounter`, `policy_drift`, `snapshot_policy`(Task 2), `PPOConfig.imitation_sigma`(Task 3), `judge`·`major_total`(Task 1)
- Produces:
  - 새 인자 `--imitation-sigma {learn,detach}`(기본 `learn`) — `_build_cfg` 가 `PPOConfig` 로 넘긴다
  - `log.jsonl` 의 **모든 줄**에 `rollout_return_mean`·`rollout_return_n`·`rollout_len_mean`·`drift_l2`·`drift_rel` 이 들어간다
  - **평가 줄**에는 그 구간의 종료 사유 집계(`outcome_goal` 등)가 더해진다
  - 요약 JSON 에 `verdict` 가 들어간다 — `judge()` 가 낸 네 줄을 `[{"name":…, "ok":…, "line":…}, …]` 로
- **새로 생기는 비용을 알고 시작해라:** `train_ppo.py` 는 지금 **선생님을 평가하지 않는다.** 판정에는 선생님 점수가 필요하므로 마지막 전체 평가 때 `evaluate_teacher(boards, seeds=(0,))` 를 단계마다 한 번씩 더 돌려야 한다. 선생님은 규칙 스택이라 느리다(442 걸음/s) — 판 12 개 × 약 3500 걸음이면 **약 95 초**다. 3M 실행이 17 분이니 감당되지만, **`--smoke` 에서는 각 단계 코스 A 한 판으로 줄여라**(안 그러면 14 초짜리 연습 모드가 100 초가 된다).
  - 이 값을 매 실행 다시 재는 이유: 시드 스윕이 "이 설정이 3 개 시드 중 몇 개에서 목표를 채웠나" 를 셀 수 있어야 하고, 그게 M4b 가 답하려는 질문이다.
- **주의:** `ReturnTracker.reset()` 은 롤아웃마다 부른다(그 롤아웃의 리턴만 보려는 것). 진행 중 누적은 클래스가 알아서 이어 간다. `OutcomeCounter` 는 **평가 때마다** 새로 만든다(구간별 집계).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_train_ppo.py` 에 더한다:
```python
@pytest.mark.slow
def test_연습_모드가_계측과_판정을_남긴다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                          "--imitation-sigma", "detach"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])

    assert summary["hparams"]["imitation_sigma"] == "detach"
    assert len(summary["verdict"]) == 4
    assert {v["name"] for v in summary["verdict"]} == {
        "완주율", "선생님 대비 점수", "출발점 대비 점수", "전 항목 중대 위반"}
    assert all(isinstance(v["ok"], bool) and v["line"] for v in summary["verdict"])

    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl", encoding="utf-8")]
    assert rows, "log.jsonl 이 비어 있다"
    for r in rows:
        for k in ("rollout_return_n", "drift_l2", "drift_rel"):
            assert k in r, k
    assert any(r["rollout_return_n"] > 0 for r in rows), "끝난 판이 한 번도 안 잡혔다"
    # 드리프트는 처음엔 0 에 가깝고 학습이 돌수록 커진다
    assert rows[0]["drift_l2"] < rows[-1]["drift_l2"]
    evals = [r for r in rows if "stages" in r]
    assert evals and any(k.startswith("outcome_") for k in evals[-1])


def test_시그마_모드가_cfg에_닿는다():
    mod = _load_train_ppo_module()   # 이 파일이 이미 쓰는 importlib 헬퍼(이름 그대로)
    a = mod._build_parser().parse_args(["--out", "x", "--imitation-sigma", "detach"])
    assert mod._build_cfg(a).imitation_sigma == "detach"
    b = mod._build_parser().parse_args(["--out", "x"])
    assert mod._build_cfg(b).imitation_sigma == "learn"
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_train_ppo.py -q -k 시그마`
Expected: FAIL — `argument --imitation-sigma: unrecognized arguments`

- [ ] **Step 3: 구현**

`scripts/train_ppo.py` 를 고친다:

1. 임포트에 더한다.
```python
from vtd_rl.eval.verdict import judge, major_total
from vtd_rl.policy.evaluate import violation_counts
from vtd_rl.rl.diagnostics import OutcomeCounter, ReturnTracker, policy_drift, snapshot_policy
```

2. `_build_parser` 에 인자를 더한다(다른 하이퍼파라미터 인자들 옆).
```python
    ap.add_argument("--imitation-sigma", choices=("learn", "detach"),
                    default=default_cfg.imitation_sigma,
                    help="detach 면 모방 손실이 σ(log_std)를 안 건드린다 — 평균은 그대로 배운다."
                         " M4a 에서 모방이 σ 를 하한에 붙박아 조향 탐색이 없었다")
```

3. `_build_cfg` 의 `dataclasses.replace(...)` 에 `imitation_sigma=a.imitation_sigma` 를 더한다.

4. 롤아웃 루프에 계측을 건다. `venv` 를 만든 뒤 `ref_state = snapshot_policy(net)` 와 `tracker = ReturnTracker(a.envs)` 를 만들고, 롤아웃 시작마다 `tracker.reset()`, `venv.step` 뒤에 한 줄 더한다:
```python
            tracker.add(reward, np.asarray(term) | np.asarray(trunc))
```
그리고 그 롤아웃의 `outcomes.add(info)` 도 함께 부른다(`outcomes` 는 평가마다 새로 만든 `OutcomeCounter`).

5. 로그 행을 만들 때 `**tracker.stats()`, `**policy_drift(net, ref_state)` 를 펼쳐 넣는다. **평가 줄에만** `**outcomes.stats()` 를 더하고, 그 뒤 `outcomes = OutcomeCounter()` 로 새로 만든다.
   - **키 충돌에 주의해라.** M4a 에서 `**stats` 가 바깥 카운터의 `updates` 를 조용히 덮어썼다. 펼쳐 넣는 모든 dict 의 키를 서로 대조해 보고, 겹치면 바깥 이름을 바꿔라.

6. 마지막 전체 평가 뒤 판정을 만들어 요약에 넣는다:
```python
    majors = {k: major_total(violation_counts(v)) for k, v in final_ev.items()}
    verdict = judge(final_ev, teacher_ev, majors, eval_seeds=a.final_eval_seeds)
    summary["verdict"] = [{"name": v.name, "ok": v.ok, "line": v.line} for v in verdict]
```
`teacher_ev` 는 단계마다 `evaluate_teacher(boards, seeds=(0,))` 로 따로 부른다 — **단계 ①②의 판을 합쳐 한 번에 넘기면 판 이름이 같아 `AssertionError: 세계 캐시가 다른 판을 준다` 로 죽는다.** 평가용 판은 `load_curriculum` 으로 직접 읽어라(`vec_env._boards()` 는 이름에 `#stage` 접미사를 붙인다).

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

그리고 **연습 모드를 실제로 돌려 `log.jsonl` 한 줄을 눈으로 읽어라.** M4a 에서 테스트는 통과하는데 로그 숫자가 틀린 버그(키 충돌)를 그렇게 잡았다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/train_ppo.py tests/rl/test_train_ppo.py
git commit -m "M4b — 학습 스크립트에 롤아웃 리턴·종료 사유·드리프트 계측과 σ 분리 모드, 판정 요약"
```

---

### Task 5: 시드 스윕 러너

**Files:**
- Create: `scripts/sweep_ppo.py`, `tests/rl/test_sweep_ppo.py`

**Interfaces:**
- Consumes: `scripts/train_ppo.py`(하위 프로세스로 부른다)
- Produces:
  - `scripts/sweep_ppo.py` — 인자 `--out-root runs/omen/<날짜>-sweep --seeds 0 1 2 --name <이름> [train_ppo 로 넘길 나머지 인자들...]`
    - 시드마다 `train_ppo.py` 를 **순차로** 부르고(동시에 돌리면 30 환경씩 경합한다) `<out-root>/<name>-s<seed>/` 에 넣는다
    - 각 실행의 요약 JSON 을 모아 `<out-root>/sweep.json` 에 쓴다: `{"name":…, "runs": [{"seed":…, "summary": {…}}, …], "spread": {…}}`
    - `spread` 는 단계별 `goal_rate`·`mean_score` 의 `{"min":…, "max":…, "mean":…}`
    - 마지막 줄에 요약 JSON 한 줄을 찍는다
    - 하나라도 실패하면 **거기서 멈추고** 0 이 아닌 코드로 끝낸다(조용히 절반만 도는 것보다 낫다)
- **왜 필요한가:** M4a 는 설정마다 시드 하나였다. M3 에서 같은 명령이 완주율 0% 와 100% 로 갈린 전례가 있어, 지금 숫자들이 실행 간 편차 안인지 밖인지 모른다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_sweep_ppo.py`
```python
import json
import os
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.mark.slow
def test_시드_두_개를_돌리고_편차를_모은다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "sweep_ppo.py"),
                          "--out-root", str(tmp_path / "sweep"), "--name", "smoke",
                          "--seeds", "0", "1", "--", "--smoke"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(summary["runs"]) == 2
    assert {r["seed"] for r in summary["runs"]} == {0, 1}
    for stage in ("stage1", "stage2"):
        sp = summary["spread"][stage]["mean_score"]
        assert sp["min"] <= sp["mean"] <= sp["max"]
    assert os.path.exists(tmp_path / "sweep" / "sweep.json")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s0" / "log.jsonl")
    assert os.path.exists(tmp_path / "sweep" / "smoke-s1" / "log.jsonl")


@pytest.mark.slow
def test_한_실행이_실패하면_멈춘다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # --steps 0 은 train_ppo 가 거부한다 → 첫 시드에서 멈춰야 한다
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "sweep_ppo.py"),
                          "--out-root", str(tmp_path / "sweep"), "--name", "bad",
                          "--seeds", "0", "1", "--", "--smoke", "--steps", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    assert out.returncode != 0
    assert not os.path.exists(tmp_path / "sweep" / "bad-s1")
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_sweep_ppo.py -q`
Expected: FAIL — `can't open file 'scripts/sweep_ppo.py'`

- [ ] **Step 3: 구현**

`scripts/sweep_ppo.py` 를 쓴다. 뼈대:
```python
"""같은 설정을 시드 여러 개로 돌려 실행 간 편차를 잰다.

M4a 는 설정마다 시드 하나였다. M3 에서 같은 명령이 완주율 0% 와 100% 로 갈린 전례가 있어
(docs/reports/m3-dagger.md), 한 번의 숫자가 편차 안인지 밖인지 알 수 없었다.
시드는 **순차로** 돈다 — 30 환경짜리 실행 둘을 동시에 띄우면 코어가 경합한다.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_ppo.py")


def _spread(values):
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="`--` 뒤의 인자를 train_ppo.py 로 그대로 넘긴다")
    a = ap.parse_args()
    extra = [x for x in a.rest if x != "--"]

    runs = []
    for seed in a.seeds:
        out = os.path.join(a.out_root, f"{a.name}-s{seed}")
        cmd = [sys.executable, TRAIN, "--out", out, "--seed", str(seed)] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            sys.exit(f"시드 {seed} 실행이 실패했다(rc={proc.returncode}) — 여기서 멈춘다")
        runs.append({"seed": seed, "summary": json.loads(proc.stdout.strip().splitlines()[-1])})

    spread = {}
    for stage in ("stage1", "stage2"):
        spread[stage] = {k: _spread([r["summary"]["stages"][stage][k] for r in runs])
                         for k in ("goal_rate", "mean_score")}
    summary = {"name": a.name, "seeds": a.seeds, "runs": runs, "spread": spread}
    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add scripts/sweep_ppo.py tests/rl/test_sweep_ppo.py
git commit -m "M4b — 시드 스윕 러너와 실행 간 편차 집계"
```

---

### Task 6: 성적표 생성기를 새 판정·계측에 맞춘다

**Files:**
- Modify: `scripts/report_m4a.py`
- Test: `tests/rl/test_report_m4a.py`(추가·수정)

**Interfaces:**
- Consumes: `judge`·`major_total`(Task 1), `--sweep` 로 읽는 `sweep.json`(Task 5)
- Produces:
  - 목표 판정을 **`verdict.judge()` 에 위임한다.** `ITEM2_MAJOR_MAX`·`ITEM7_MAJOR_MAX`·`_item_goal_line` 은 **지운다**(복사본을 남기지 마라)
  - 항목별 표에 **중대 합계 행**을 더한다(M3·학생·선생님 각각)
  - 학습 곡선 표에 `rollout_return_mean`·`drift_rel` 열을 더한다
  - 새 인자 `--sweep <sweep.json>`(여러 번) — 시드 편차 표를 만든다: 실행 이름 · 시드 수 · 단계별 `goal_rate`·`mean_score` 의 min~max
  - 파일 이름은 `--out` 이 정한다(M4b 는 `docs/reports/m4b-ppo.md`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_report_m4a.py` 에 더한다:
```python
def test_항목_문턱_상수가_사라졌다():
    src = open(os.path.join(REPO, "scripts", "report_m4a.py"), encoding="utf-8").read()
    assert "ITEM7_MAJOR_MAX" not in src and "ITEM2_MAJOR_MAX" not in src
    assert "_item_goal_line" not in src
    assert "from vtd_rl.eval.verdict import" in src


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
```

- [ ] **Step 2: 실패 확인**

Run: `env -u PYTHONPATH .venv/bin/pytest tests/rl/test_report_m4a.py -q -k 문턱`
Expected: FAIL — `assert "ITEM7_MAJOR_MAX" not in src`

- [ ] **Step 3: 구현**

`scripts/report_m4a.py` 를 고친다:
- `ITEM2_MAJOR_MAX`·`ITEM7_MAJOR_MAX`·`_item_goal_line` 을 지우고 `from vtd_rl.eval.verdict import judge, major_total` 로 바꾼다.
- `_goal_lines` 는 지금 시그니처가 `(m4a_ev, teacher_ev, rows, best_step, eval_seeds)`(`scripts/report_m4a.py:303`, 호출부 `:471`)다. **`major_totals` 를 받도록 인자를 하나 더한다** — 호출부에서 `{k: major_total(violation_counts(v)) for k, v in m4a_ev.items()}` 로 만들어 넘겨라. 본문은 `judge()` 가 낸 `GoalVerdict` 목록을 `- ` 줄로 옮기고, 맨 위에 "목표 4 줄 중 **N 줄 달성**"(보류가 있으면 그 수도) 을 붙인다. `_best_step_caveat` 은 그대로 둔다.
- 항목별 표 마지막에 **중대 합계** 행을 더한다(`major_total()` 사용).
- 곡선 표에 두 열을 더한다. 값이 `None` 이면 `—` 로 찍어라(끝난 판이 없던 롤아웃이 있다).
- `--sweep` 을 `action="append"` 로 받아 시드 편차 표를 만든다.

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q; echo rc=$?`
Expected: `rc=0`

- [ ] **Step 5: 커밋**

```bash
git add scripts/report_m4a.py tests/rl/test_report_m4a.py
git commit -m "M4b 성적표 — 판정을 verdict 로 위임, 중대 합계·계측 열·시드 편차"
```

---

### Task 7: 본 실험과 성적표 (M4b 완료 증거)

**Files:**
- Create: `docs/reports/m4b-ppo.md`(스크립트가 생성), `docs/reports/m4b-ppo-notes.md`(사람이 쓰는 해석)
- Modify: `README.md`

**Interfaces:**
- 돌릴 설정 **세 개**(전부 OMEN 30 환경·3M 스텝·`--device cpu`·M3 학생에서 출발, **시드 0·1·2**):
  - `base` — M4a run1 과 같은 기본값. **시드 편차의 기준선**이다
  - `sigma-detach` — `--imitation-sigma detach`. 이 계획의 핵심 가설
  - `sigma-detach-ent` — `--imitation-sigma detach --entropy-coef 0.05`. σ 를 풀고 엔트로피도 밀면 달라지는지
- 실행은 시드마다 순차라 설정 하나에 약 **55 분**(3M 스텝 ≈ 17 분 × 3), 셋이면 약 **3 시간**이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rl/test_report_m4a.py` 의 Task 6 테스트가 이 작업의 산출물 형식을 이미 잠근다. **이 작업은 새 테스트를 만들지 않는다** — 실행과 성적표 작성이다.

- [ ] **Step 2: 실행 전 확인**

OMEN 에 코드를 올리고 빠른 스위트가 도는지 본다.
```bash
for d in vtd_rl scripts tests; do
  rsync -a --delete --exclude '__pycache__' $d/ 100.87.135.28:vtd-rl-urban-driving/$d/
done
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && env -u PYTHONPATH .venv/bin/pytest -q -m "not slow"; echo rc=$?'
```
Expected: `rc=0`

- [ ] **Step 3: 세 설정을 돌린다 (OMEN)**

**앞단에서** 돌린다. 설정 하나씩, 끝날 때까지 기다린다.
```bash
ssh 100.87.135.28 'cd ~/vtd-rl-urban-driving && nice -n 5 env -u PYTHONPATH .venv/bin/python scripts/sweep_ppo.py \
  --out-root runs/omen/2026-09-22-m4b --name base --seeds 0 1 2 -- \
  --init runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
  --dagger-data runs/lab-main/2026-09-17-dagger-fix2/data \
  --envs 30 --steps 3000000 --device cpu' | tail -c 600
```
`--name sigma-detach` 에는 `--imitation-sigma detach` 를, `--name sigma-detach-ent` 에는 거기에 `--entropy-coef 0.05` 를 더해 같은 방식으로 돌린다.

결과를 회수한다.
```bash
rsync -a 100.87.135.28:vtd-rl-urban-driving/runs/omen/2026-09-22-m4b/ runs/omen/2026-09-22-m4b/
```

- [ ] **Step 4: 해석 노트를 쓰고 성적표를 만든다**

`docs/reports/m4b-ppo-notes.md` 를 쓴다. **숫자는 성적표가 생성하니 여기엔 해석만 쓴다.** 반드시 담을 것:
- 목표 네 줄 각각의 결과와, 못 채운 줄에 대해 **무엇이 원인으로 보이는지와 그 근거**
- **σ 분리가 실제로 σ 를 풀었나** — `log_std[0]` 곡선을 base 와 비교해서
- **붕괴가 여전한가** — 그렇다면 종료 사유 집계와 `drift_rel` 이 무엇을 말하는가. M4a 는 이 두 계측이 없어 붕괴를 규명하지 못했다
- **롤아웃 리턴이 평가 보상과 같은 방향으로 움직이나** — 다르면 "학습은 되는데 평가가 나쁘다" 는 뜻이고, 그건 M4a 가 배제하지 못했던 가설이다
- 시드 세 개의 편차. M4a 숫자들이 편차 안이었는지
- **이 성적표가 주장하지 않는 것**(설정 3 개·시드 3 개·커리큘럼은 단계 ①② 뿐)

성적표를 만든다.
```bash
env -u PYTHONPATH .venv/bin/python scripts/report_m4a.py \
  --run runs/omen/2026-09-22-m4b/sigma-detach-s0 \
  --compare runs/omen/2026-09-22-m4b/base-s0 \
  --compare runs/omen/2026-09-22-m4b/sigma-detach-ent-s0 \
  --sweep runs/omen/2026-09-22-m4b/sweep.json \
  --m3 runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
  --notes docs/reports/m4b-ppo-notes.md \
  --out docs/reports/m4b-ppo.md --eval-seeds 3 --device cpu
```
(스윕이 설정마다 `sweep.json` 을 남기므로 `--sweep` 을 셋 다 준다. `--run` 은 **목표를 가장 잘 맞춘 설정**으로 고르되, **그렇게 고른 사실을 노트에 적어라** — M4a 에서 그 고지 없이 고르면 결론이 왜곡된다는 것을 배웠다.)

**목표를 못 채우면 기준을 고치지 마라.** 숫자와 관찰을 적고, 무엇을 더 봐야 하는지 남긴다.

- [ ] **Step 5: README 와 커밋**

`README.md` 의 PPO 단락에 M4b 결과 링크를 더한다:
```markdown
`vtd_rl/eval/verdict.py` 가 목표 판정을 한 곳에 모으고, `vtd_rl/rl/diagnostics.py` 가 롤아웃
리턴·종료 사유·정책 드리프트를 남긴다. 시드 여러 개 실행은 `scripts/sweep_ppo.py`.
결과: [docs/reports/m4b-ppo.md](docs/reports/m4b-ppo.md)
```

```bash
git add docs/reports/m4b-ppo.md docs/reports/m4b-ppo-notes.md README.md
git commit -m "M4b 증거 — σ 분리·계측·시드 스윕 결과와 목표 판정"
```

---

## 이 계획이 미루는 것

- **커리큘럼 단계 ③④⑤**(사물·정지차 / 보행자·교통 / 연습코스 전체)와 자동 진급. **PPO 가 단계 ①② 에서 출발점을 못 넘는 동안 판을 어렵게 만들면 실패 원인만 늘어난다** — 목표 3 번(출발점 대비 점수)을 채운 뒤에 한다.
- 두 머신에서 **동시에** 다른 설정을 돌리고 결과를 모으기(스펙 §6.5). 지금은 OMEN 순차로 충분하다.
- 학습률·클립·롤아웃 길이 탐색.
- M4a 최종 리뷰가 M4b 로 넘긴 성적표 결함 둘: `_detect_best_step` 이 중간 체크포인트를 못 찾으면 "사실상 M3 워밍스타트" 경고가 **조용히 사라지는 것**(그리고 사유 문자열이 틀리게 나오는 것), 그리고 라이브 평가(시드 N)와 주기 평가(시드 1) 값이 **같은 열**에 섞이는 것.
- `scripts/run_dagger.py:357,359` 의 낡은 "남은 숙제" 문구(M4a 가 고친 항목을 아직 숙제로 적고 있다).
