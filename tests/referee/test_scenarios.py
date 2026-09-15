import pytest

from vtd_rl.referee.oracle import score_episode
from vtd_rl.referee.scenarios import SCENARIOS, episode_rows
from vtd_rl.rollout import run_episode


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_대본_판은_채점기에서_기대한_항목을_일으킨다(name):
    sc, rows = episode_rows(name)
    fired = {i for _, i, _ in score_episode(rows, sc.board, sections=5, use_map=True).hits}
    assert sc.expect <= fired, f"{name}: 기대 {sorted(sc.expect)} / 채점기 {sorted(fired)}"
    assert not (sc.forbid & fired), f"{name}: 금지 {sorted(sc.forbid)} / 채점기 {sorted(fired)}"


def _stop_span(rows, t_min=2.0, v_max=1 / 3.6):
    """t_min 이후 v<=v_max 인 연속 구간 중 가장 긴 길이[s]."""
    best = span = 0.0
    start = None
    for r in rows:
        if r["t"] > t_min and r["v"] <= v_max:
            if start is None:
                start = r["t"]
            span = r["t"] - start
            best = max(best, span)
        else:
            start = None
    return best


def test_make_driver_를_두_번_불러도_StopAt_이_다시_선다():
    """episode_rows 는 Scenario 를 캐시하므로, 진단 때처럼 같은 Scenario 에
    make_driver 를 두 번 불러도(brief 의 `run_episode(sc.board, sc.make_driver(sc.board))`
    패턴) StopAt 이 소진된 상태를 재사용해 두 번째 판이 서지 않으면 안 된다."""
    sc = SCENARIOS["red_far_stopper"]()
    run1 = run_episode(sc.board, sc.make_driver(sc.board))
    run2 = run_episode(sc.board, sc.make_driver(sc.board))
    assert _stop_span(run1.rows) > 2.5, "1 번째 실행이 서지 않았다"
    assert _stop_span(run2.rows) > 2.5, "2 번째 실행이 서지 않았다(StopAt 재사용 버그)"
    assert len(run1.rows) == len(run2.rows)
