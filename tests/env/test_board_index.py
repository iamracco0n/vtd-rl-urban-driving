import math

from vtd_rl.env.board_index import board_index, clear_board_index_cache
from vtd_rl.world.board import load_board, slice_board

H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}


def h_slice():
    return slice_board(load_board(H), 0.0, 250.0, "H_0_250")


def test_코스_H_앞쪽_거리():
    bi = board_index(h_slice())
    assert bi.signal_s and abs(bi.signal_s[0] - 96.3) < 2.0
    assert [s.s for s in bi.signals] and min(s.s for s in bi.signals) == bi.signal_s[0]
    assert any(abs(s - 104.5) < 2.0 for s in bi.crosswalk_s)
    assert bi.ahead(0.0, "signal") == bi.signal_s[0]
    assert bi.ahead(0.0, "crosswalk") > bi.ahead(110.0, "crosswalk")
    assert bi.ahead(9e9, "signal") == math.inf
    assert bi.plan_at(100).get("lim") == 13.889      # 코스 H 제한 50 km/h
    assert bi.plan_at(10 ** 9) == {}


def test_캐시는_같은_색인을_준다():
    clear_board_index_cache()
    b = h_slice()
    assert board_index(b) is board_index(b)
    clear_board_index_cache()
    assert board_index(b) is not None
