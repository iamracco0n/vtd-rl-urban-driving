import math

from vtd_rl import rule_stack as rs
from vtd_rl.referee.rows import RowRecorder


def state_and_cmd():
    s = rs.State(x=10.0, y=-2.0, heading=0.3, speed=4.25, tl_id=151, tl_state=rs.TL_GREEN,
                 objects=[rs.Obj(7, 20.0, -1.0, 0.0, 0.1, 1.3, 0.6, 0.7, 1.8)])
    c = rs.Command(steer=0.01, accel=0.5, turn=rs.TS_LEFT, reason="SCRIPT", cap_by="SCRIPT",
                   d_ego=0.42)
    return s, c


def test_행_사전은_채점기_load_와_같다(tmp_path):
    s, c = state_and_cmd()
    path = tmp_path / "run.csv"
    rec = RowRecorder(str(path))
    rows = [rec.record(1.25, s, c), rec.record(1.30, s, c)]
    rec.close()
    assert rs.score_fma.load(str(path)) == rows
    row = rows[0]
    assert row["sig"] == rs.TS_LEFT and row["tid"] == 151 and math.isclose(row["d_ego"], 0.42)
    assert row["objs"].count(":") >= 9 and row["clr"] < 99.9


def test_파일_없이도_행을_만든다():
    s, c = state_and_cmd()
    row = RowRecorder().record(0.0, s, c)
    assert set(row) == {"t", "x", "y", "h", "v", "tl", "tid", "reason", "sig", "clr",
                        "d_ego", "objs", "cap_by"}
