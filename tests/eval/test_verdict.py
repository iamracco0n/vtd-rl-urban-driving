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
