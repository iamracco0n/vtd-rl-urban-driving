"""목표 판정 — 학습 스크립트와 성적표가 **같은 규칙**을 쓰게 한 곳에 모은다.

M4a 는 판정이 성적표에만 있었고, 목표를 **항목 두 개**(⑦ 적색 정지·② 보호구역)의 절대 건수로
걸었다. 그 결과 run2 가 항목 ⑦ 을 64 → 19 로 줄이면서 항목 ③(차로 유지) 중대를 0 → 18 로
터뜨렸는데도 판정은 "달성" 이 나왔다 — 총점은 94.2 → 90.7 로 내려갔는데도. 풍선의 한쪽을
눌러 다른 쪽을 부풀리는 정책이 통과한 것이다.

그래서 스펙 §6.4 의 원래 문구(완주율과 **총점**)로 되돌리고, M4a 가 묻지 않았던 두 가지를
더한다 — **출발점(M3 학생)보다 나은가**, 그리고 **중대 위반 총합이 늘지 않았는가**.
항목별 건수는 진단으로만 본다.

Task 1 첫 리뷰에서 두 가지가 더 드러났다.

1. `mean_score` 는 미완주 판을 0 점으로 치므로 항상 `mean_score <= 100 * goal_rate` 다.
   그래서 여유폭 없이 "3번(출발점 대비 점수) 이 M3 보다 크면" 만 걸면, 완주율(1번)과
   선생님 대비 점수(2번)는 3번이 참일 때 **저절로** 참이 된다 — 실제로 깨질 수 있는 줄은
   3·4번뿐인데 "4줄 중 2줄 달성" 처럼 부풀려 보인다. 그래서 1·2번은
   `GoalVerdict.precondition=True` 로 표시해 3번에 이미 포함된 전제일 뿐 독립 목표가
   아님을 드러낸다. 또한 평가판이 고정된 같은 18판이고 M4a 는 결과를 본 뒤 가장 잘 맞춘
   실행을 골랐으므로, 여유폭이 0이면 "여러 번 돌려 운 좋게 넘긴 값" 도 통과한다 —
   `M3_SCORE_MARGIN` 으로 막는다.
2. 미완주 판은 도달하지 못한 구간의 위반이 채점표에 아예 없어 중대 위반이 실제보다 적게
   잡힌다. 점수는 미완주를 0점으로 **벌하는데** 위반 집계는 반대로 **상**을 주는 셈이다.
   그래서 중대 위반은 완주한 판만(`completed_only`) 골라 센 뒤 합산한다.

이 두 번째 수정 자체가 새 구멍을 냈다(2026-09-22, 전체 브랜치 리뷰). `completed_only` 는
"3번(출발점 대비 점수)이 완주를 사실상 강제하니 안전하다" 고 판단했는데 틀렸다 — 4번은 3번과
**독립으로** 인쇄되고 판정된다. `sigma-detach-ent-s2` 실행(9회 중 가장 붕괴, 점수 32.4, 완주율
33%)이 실측으로 증명했다: 완주 판이 18개 중 6개뿐이라 중대 절대 합계(9, 18)가 M3 의 18판 기준
합계(15, 79)보다 작아져 4번이 "달성" 으로 나왔다 — **판을 덜 끝낼수록 통과하는**, M4a 를 망친
것과 같은 종류의 풍선 누르기다. 그래서 완주 판 수가 M3 기준(`M3_COMPLETED`, 18판)과 다르면
4번을 **보류**한다 — 절대 합계를 다른 분모로 비교하지 않는다.
"""
from dataclasses import dataclass

GOAL_RATE_MIN = 0.9
TEACHER_SCORE_SLACK = 10.0

# mean_score 는 미완주 판을 0점으로 치므로 항상 mean_score <= 100 * goal_rate 다. 여유폭 없이
# "M3 보다 크면" 만 걸면 3번이 참인 순간 1·2번도 저절로 참이 돼(위 모듈 docstring 참고),
# 게다가 그 문턱을 ε 만 넘겨도(운 좋게 한 번 잘 나온 값도) 통과해 버린다. 단계② 중대 위반
# 슬롯 7~8개에 해당하는 크기(0.5점)를 여유로 둔다 — 슬롯 하나(≈0.067점)와는 확실히 구분되는
# 크기다.
M3_SCORE_MARGIN = 0.5

# 2026-09-21 실측(시드 3개, `runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt`).
# 단계① 중대: 항목② 15. 단계② 중대: 항목② 15 + 항목⑦ 64 = 79.
M3_SCORE = {"stage1": 98.5, "stage2": 94.2}
M3_MAJOR_TOTAL = {"stage1": 15, "stage2": 79}

# M3 기준값이 시드 3개 합계라, 시드 수가 다르면 3·4번은 비교 자체가 성립하지 않는다.
REQUIRED_EVAL_SEEDS = 3

# M3_MAJOR_TOTAL 은 완주한 판 18개(단계별 시드 3개 x 보드 6개, M3 학생이 두 단계 모두
# 100% 완주해 총 판 수와 완주 판 수가 같다) 를 합산한 값이다. `completed_only` 는 조기
# 종료한 판을 걸러내므로, 지금 평가의 완주 판 수가 18이 아니면 절대 합계 자체가 다른
# 분모에서 나온 것이라 그대로 비교할 수 없다 — 완주 판이 적을수록 위반도 적게 잡혀
# "덜 끝낼수록 통과" 하는 풍선 누르기가 된다(2026-09-22 실측 `sigma-detach-ent-s2`).
M3_COMPLETED = {"stage1": 18, "stage2": 18}


@dataclass(frozen=True)
class GoalVerdict:
    name: str
    ok: bool
    line: str
    # True 면 이 줄은 3번(출발점 대비 점수)에 이미 포함된 전제 조건일 뿐 독립 목표가
    # 아니다(위 모듈 docstring 1번). 헤드라인 "N줄 중 M줄 달성" 은 이 플래그가 False 인
    # 줄만 세야 한다.
    precondition: bool = False


def major_total(counts: dict) -> int:
    """`violation_counts()` 결과에서 **중대만** 합산한다 — 경미는 감점이 작아 총점에 묻힌다."""
    return sum(v.get("major", 0) for v in counts.values())


def completed_only(ev: dict) -> dict:
    """완주한 판만 남긴 같은 모양의 dict.

    미완주 판은 도달하지 못한 구간의 슬롯이 채점표에 아예 없어 위반이 실제보다 적게 잡힌다.
    `major_total(violation_counts(ev))` 앞에 이걸 끼워 완주한 판만 세게 한다.
    """
    return {**ev, "episodes": [e for e in ev["episodes"] if e.outcome == "goal"]}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _label(ok: bool, held: bool) -> str:
    """보류일 때는 "미달" 이라고 쓰지 않는다 — 판정 자체가 유보된 것이지 실패가 아니다."""
    if held:
        return "보류"
    return "달성" if ok else "미달"


def _completed_count(ev: dict):
    """`ev["episodes"]` 중 완주(outcome == "goal") 개수. 정보가 없으면 None."""
    episodes = ev.get("episodes")
    if not episodes:
        return None
    return sum(1 for e in episodes if e.outcome == "goal")


def _completion_note(ev: dict):
    """`ev["episodes"]` 에서 완주 수/전체 수 "N/M" 문자열을 낸다. 정보가 없으면 None."""
    episodes = ev.get("episodes")
    if not episodes:
        return None
    return f"{_completed_count(ev)}/{len(episodes)}"


def judge(student: dict, teacher: dict, major_totals: dict, eval_seeds: int) -> list[GoalVerdict]:
    missing = [stage for stage in ("stage1", "stage2") if stage not in student]
    if missing:
        raise ValueError(f"student 평가 결과에 {', '.join(missing)} 가 없다 — "
                          "두 단계를 모두 평가해야 판정할 수 있다")

    s1, s2 = student["stage1"], student["stage2"]
    t1, t2 = teacher["stage1"], teacher["stage2"]
    held = eval_seeds != REQUIRED_EVAL_SEEDS
    hold = (f" (M3 기준값이 시드 {REQUIRED_EVAL_SEEDS}개 합계인데 이번 평가는"
            f" 시드 {eval_seeds}개다. 같은 시드 수로 다시 재야 비교가 성립한다)")

    goal_ok = s1["goal_rate"] >= GOAL_RATE_MIN and s2["goal_rate"] >= GOAL_RATE_MIN
    score_ok = (s1["mean_score"] >= t1["mean_score"] - TEACHER_SCORE_SLACK
                and s2["mean_score"] >= t2["mean_score"] - TEACHER_SCORE_SLACK)
    beats_m3 = (s1["mean_score"] > M3_SCORE["stage1"] + M3_SCORE_MARGIN
                and s2["mean_score"] > M3_SCORE["stage2"] + M3_SCORE_MARGIN)
    majors_ok = (major_totals["stage1"] <= M3_MAJOR_TOTAL["stage1"]
                 and major_totals["stage2"] <= M3_MAJOR_TOTAL["stage2"])

    n1, n2 = _completion_note(s1), _completion_note(s2)
    completion_suffix = (f" (완주 판 기준 — 단계① {n1}, 단계② {n2})"
                         if n1 is not None and n2 is not None else "")

    # 4번만의 추가 보류 사유: 완주 판 수가 M3 기준(18판)과 다르면 절대 합계를 비교할 수
    # 없다(위 모듈 docstring 참고). episodes 정보가 없으면(합성 테스트 등) 판단을 보류하지
    # 않는다 — 알 수 없는 것과 다른 것은 다르다.
    c1, c2 = _completed_count(s1), _completed_count(s2)
    denom_parts = []
    if c1 is not None and c1 != M3_COMPLETED["stage1"]:
        denom_parts.append(f"단계① 완주 {c1}/{M3_COMPLETED['stage1']}판")
    if c2 is not None and c2 != M3_COMPLETED["stage2"]:
        denom_parts.append(f"단계② 완주 {c2}/{M3_COMPLETED['stage2']}판")
    denom_mismatch = bool(denom_parts)
    denom_hold = (f" (보류 — {', '.join(denom_parts)}이라 M3 기준(18판) 합계와 절대건수로"
                  " 비교할 수 없다)") if denom_mismatch else ""
    held4 = held or denom_mismatch

    return [
        GoalVerdict("완주율", goal_ok,
                    f"완주율 단계①② 모두 ≥{_pct(GOAL_RATE_MIN)} → **{'달성' if goal_ok else '미달'}**"
                    f" (실측: 단계① {_pct(s1['goal_rate'])}, 단계② {_pct(s2['goal_rate'])})",
                    precondition=True),
        GoalVerdict("선생님 대비 점수", score_ok,
                    f"점수가 선생님보다 {TEACHER_SCORE_SLACK:.0f}점 넘게 낮지 않을 것 →"
                    f" **{'달성' if score_ok else '미달'}** (실측: 단계① {s1['mean_score']:.1f} vs"
                    f" 선생님 {t1['mean_score']:.1f}, 단계② {s2['mean_score']:.1f} vs"
                    f" 선생님 {t2['mean_score']:.1f})",
                    precondition=True),
        GoalVerdict("출발점 대비 점수", beats_m3 and not held,
                    f"점수가 출발점(M3 학생)보다 {M3_SCORE_MARGIN:.1f}점 넘게 나을 것 →"
                    f" **{_label(beats_m3, held)}** (실측: 단계①"
                    f" {s1['mean_score']:.1f} vs M3 {M3_SCORE['stage1']:.1f}, 단계②"
                    f" {s2['mean_score']:.1f} vs M3 {M3_SCORE['stage2']:.1f})"
                    + (hold if held else "")),
        GoalVerdict("전 항목 중대 위반", majors_ok and not held4,
                    f"중대 위반 **합계**가 M3 학생보다 늘지 않을 것 →"
                    f" **{_label(majors_ok, held4)}** (실측: 단계①"
                    f" {major_totals['stage1']} vs M3 {M3_MAJOR_TOTAL['stage1']}, 단계②"
                    f" {major_totals['stage2']} vs M3 {M3_MAJOR_TOTAL['stage2']})"
                    + completion_suffix
                    + (hold if held else "")
                    + denom_hold),
    ]
