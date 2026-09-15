import os
import re

from vtd_rl import rule_stack as rs


def test_서브모듈_경로():
    assert os.path.isfile(rs.path("src", "drive.py"))
    assert os.path.isfile(rs.path("routes", "HL_FMA_NEW_H.json"))


def test_서브모듈_커밋은_고정값():
    assert re.fullmatch(r"[0-9a-f]{40}", rs.commit())
    assert rs.commit() == "4a3f7037ef9ed5d7fae15107f209a86e5d550b00"


def test_규칙_스택_심볼():
    s = rs.State(x=1.0, y=2.0, heading=0.5)
    assert s.objects == [] and s.tl_state == rs.TL_UNSET
    assert rs.Command().turn == rs.TS_OFF
    assert rs.HEADER.startswith("t,x,y,heading,v,")
    assert callable(rs.mock_vtd.step_actor)


def test_지도_DB():
    db = rs.map_db()
    assert len(db["tl_map"]) == 214
    assert db["tl_map"][151] and len(db["tl_map"][151]) == 2
    assert len(db["stoplines_all"]) == 710
    assert len(db["stoplines_all"][0]) == 3
    assert isinstance(db["crosswalks"], list) and isinstance(db["stoplines"], list)
    assert rs.map_db() is db          # 두 번째는 캐시
