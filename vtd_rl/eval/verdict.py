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
