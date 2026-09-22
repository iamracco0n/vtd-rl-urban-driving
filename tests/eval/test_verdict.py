from types import SimpleNamespace

import pytest

from vtd_rl.eval.verdict import (GOAL_RATE_MIN, M3_COMPLETED, M3_MAJOR_TOTAL,
                                 M3_SCORE, M3_SCORE_MARGIN, TEACHER_SCORE_SLACK,
                                 completed_only, judge, major_total)
from vtd_rl.policy.evaluate import violation_counts


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
    # 99.1 이어야 stage1 문턱(M3 98.5 + margin 0.5 = 99.0) 을 넘는다.
    v = judge(both(1.0, 99.1), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert all(x.ok for x in v)


def test_완주율이_모자라면_첫_줄만_깨진다():
    s = {"stage1": ev(0.89, 99.1), "stage2": ev(1.0, 99.1)}
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


def test_M3보다_여유폭_안에서만_높으면_세번째가_여전히_깨진다():
    # 여유폭이 없으면 이 값(M3+0.01)이 통과해 버린다 — 운 좋게 한 번 넘긴 값도 통과하면 안 된다.
    s = {"stage1": ev(1.0, M3_SCORE["stage1"] + 0.01), "stage2": ev(1.0, M3_SCORE["stage2"] + 0.01)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[2].ok is False


def test_M3보다_여유폭_넘게_높으면_세번째가_통과한다():
    s = {"stage1": ev(1.0, M3_SCORE["stage1"] + M3_SCORE_MARGIN + 0.01),
         "stage2": ev(1.0, M3_SCORE["stage2"] + M3_SCORE_MARGIN + 0.01)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[2].ok is True


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
    assert v[2].ok is False and "보류" in v[2].line and "미달" not in v[2].line
    assert v[3].ok is False and "보류" in v[3].line and "미달" not in v[3].line


def test_문턱_상수가_스펙_값이다():
    assert GOAL_RATE_MIN == 0.9 and TEACHER_SCORE_SLACK == 10.0
    # 이 커밋의 존재 이유인 M3 기준값 자체가 안 잠겨 있으면 상수를 바꿔도 테스트가 안 잡는다.
    assert M3_SCORE == {"stage1": 98.5, "stage2": 94.2}
    assert M3_MAJOR_TOTAL == {"stage1": 15, "stage2": 79}


# --- 리뷰(C1/I3): 1·2번은 3번의 전제일 뿐 독립 목표가 아니다 --------------------------


def test_전제조건_플래그가_붙는다():
    v = judge(both(1.0, 99.1), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[0].precondition is True and v[1].precondition is True
    assert v[2].precondition is False and v[3].precondition is False


def test_3번이_참이면_1_2번도_참():
    # mean_score <= 100 * goal_rate 항등식 때문에 3번이 참이면 1·2번은 저절로 참이다.
    s = {"stage1": ev(1.0, 99.5), "stage2": ev(1.0, 95.0)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[2].ok is True
    assert v[0].ok is True and v[1].ok is True


# --- 리뷰(I4): 미완주 판은 위반 집계에서 빼야 한다 ------------------------------------


def test_completed_only는_미완주_판을_뺀다():
    raw = {"episodes": [SimpleNamespace(outcome="goal"),
                        SimpleNamespace(outcome="timeout"),
                        SimpleNamespace(outcome="goal")]}
    out = completed_only(raw)
    assert len(out["episodes"]) == 2
    assert all(e.outcome == "goal" for e in out["episodes"])


def test_completed_only가_major_total을_바꾼다():
    ep_goal = SimpleNamespace(outcome="goal", sheet=[{2: "major"}])
    ep_timeout = SimpleNamespace(outcome="timeout", sheet=[{2: "major"}, {3: "major"}])
    raw = {"episodes": [ep_goal, ep_timeout]}
    before = major_total(violation_counts(raw))
    after = major_total(violation_counts(completed_only(raw)))
    assert before == 3   # 미완주 판의 중대 2건까지 세면 3건
    assert after == 1    # 완주한 판만 세면 1건
    assert after < before


def test_네번째_줄에_완주_분모를_적는다():
    eps1 = [SimpleNamespace(outcome="goal")] * 5 + [SimpleNamespace(outcome="crash")]
    eps2 = [SimpleNamespace(outcome="goal")] * 4 + [SimpleNamespace(outcome="timeout")] * 2
    s = {"stage1": {"goal_rate": 1.0, "mean_score": 99.0, "episodes": eps1},
         "stage2": {"goal_rate": 1.0, "mean_score": 99.0, "episodes": eps2}}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert "5/6" in v[3].line
    assert "4/6" in v[3].line


# --- 리뷰(2026-09-22, 전체 브랜치): completed_only 가 4번의 분모를 M3 기준(18판)과 다르게
# --- 만들면 "덜 끝낼수록 통과" 해 버린다. 완주 판 수가 M3_COMPLETED 와 다르면 4번을 보류한다.


def test_sigma_detach_ent_s2_는_분모가_다르면_네번째가_달성이_아니다():
    """실측(`sigma-detach-ent-s2`, 9회 중 가장 붕괴): 완주 6/18 판인데 M3 중대합계(15,79)는
    18판 기준이라 절대건수를 그대로 비교하면 안 된다. 판을 덜 끝낼수록 위반도 적게 잡혀
    "달성" 이 나와 버리는 것이 이 회귀의 존재 이유다 — 이 리뷰 라운드의 핵심 계약."""
    eps1 = [SimpleNamespace(outcome="goal")] * 6 + [SimpleNamespace(outcome="crash")] * 12
    eps2 = [SimpleNamespace(outcome="goal")] * 6 + [SimpleNamespace(outcome="timeout")] * 12
    s = {"stage1": {"goal_rate": 6 / 18, "mean_score": 32.4, "episodes": eps1},
         "stage2": {"goal_rate": 6 / 18, "mean_score": 31.8, "episodes": eps2}}
    v = judge(s, TEACHER, {"stage1": 9, "stage2": 18}, eval_seeds=3)
    assert v[3].ok is False
    assert "달성" not in v[3].line
    assert "보류" in v[3].line


def test_완주_18_18이면_네번째가_평소처럼_판정된다():
    """분모 검사를 더했다고 정상 케이스(기존 테스트들)가 깨지면 안 된다."""
    eps1 = [SimpleNamespace(outcome="goal")] * M3_COMPLETED["stage1"]
    eps2 = [SimpleNamespace(outcome="goal")] * M3_COMPLETED["stage2"]
    s = {"stage1": {"goal_rate": 1.0, "mean_score": 99.1, "episodes": eps1},
         "stage2": {"goal_rate": 1.0, "mean_score": 99.1, "episodes": eps2}}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[3].ok is True
    assert "보류" not in v[3].line


# --- 리뷰(I1): major_total 은 fail-open 이면 안 된다(실제 violation_counts 와 결합) -----


def test_violation_counts와_실제로_결합해도_중대를_센다():
    """손으로 흉내 낸 dict 만으로는 producer(violation_counts)와의 키 어긋남을 못 잡는다."""
    ep1 = SimpleNamespace(sheet=[{2: "major", 3: "minor"}, {7: "major"}])
    ep2 = SimpleNamespace(sheet=[{2: "major"}, {3: "major", 7: "minor"}])
    counts = violation_counts({"episodes": [ep1, ep2]})
    assert major_total(counts) == 4


# --- 리뷰(I5): 단계①②를 바꿔치기하거나 한쪽을 무시해도 잡혀야 한다 ----------------------


def test_단계1만_통과하고_단계2는_떨어진다():
    s = {"stage1": ev(1.0, 99.1), "stage2": ev(0.5, 40.0)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[0].ok is False
    assert v[1].ok is False
    assert v[2].ok is False
    assert v[3].ok is True


def test_단계2만_통과하고_단계1은_떨어진다():
    s = {"stage1": ev(0.5, 40.0), "stage2": ev(1.0, 99.1)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[0].ok is False
    assert v[1].ok is False
    assert v[2].ok is False
    assert v[3].ok is True


def test_출발점_비교가_단계를_바꿔치기하면_안된다():
    # 단계① 96.0 은 단계①의 문턱(M3 98.5+0.5=99.0) 을 못 넘는다. 단계②의 문턱(M3
    # 94.2+0.5=94.7) 과 바꿔치기하면 96.0 > 94.7 이 참이 돼 버려 이 값이 틀렸음이 드러난다.
    s = {"stage1": ev(1.0, 96.0), "stage2": ev(1.0, 99.5)}
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[2].ok is False


def test_완주율_정확히_0_9는_통과():
    v = judge(both(0.9, 99.1), TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[0].ok is True


def test_점수가_정확히_선생님_10점_아래면_통과():
    s = {"stage1": ev(1.0, 89.6), "stage2": ev(1.0, 89.5)}   # 99.6-10, 99.5-10
    v = judge(s, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
    assert v[1].ok is True


def test_중대_합계가_M3와_정확히_같으면_네번째가_통과():
    # 4번은 "늘지 않을 것" 이므로 동률은 통과여야 한다.
    v = judge(both(1.0, 99.1), TEACHER, dict(M3_MAJOR_TOTAL), eval_seeds=3)
    assert v[3].ok is True


# --- 리뷰: student 에 단계가 없으면 KeyError 대신 ValueError -------------------------


def test_student에_단계가_없으면_ValueError():
    with pytest.raises(ValueError, match="stage2"):
        judge({"stage1": ev(1.0, 99.0)}, TEACHER, {"stage1": 10, "stage2": 50}, eval_seeds=3)
