"""주행 한 프레임 -> 채점기 행.

규칙 스택 RunLogger 가 쓰는 CSV 한 줄을 그대로 만들고, score_fma.load 와 같은 변환으로 사전을 만든다.
심판과 사후 채점기가 글자 하나 다르지 않은 입력을 받게 하려는 것이다.
"""
import csv
import io

from vtd_rl import rule_stack as rs

_FIELDS = rs.HEADER.strip().split(",")


def parse_row(line: str) -> dict:
    r = next(csv.DictReader(io.StringIO(line), fieldnames=_FIELDS))

    def g(k, d=None):                       # score_fma.load 의 g 와 같은 규칙
        v = r.get(k)
        return d if v in (None, "") else v

    return dict(
        t=float(r["t"]), x=float(r["x"]), y=float(r["y"]),
        h=float(r["heading"]), v=float(r["v"]),
        tl=int(r["tl_state"]), tid=int(r["tl_id"]),
        reason=r.get("reason", ""),
        sig=int(g("sig", 0)),
        clr=float(g("clr", 99.9)),
        d_ego=float(g("d_ego", 0.0)),
        objs=r.get("objs", ""),
        cap_by=r.get("cap_by", ""))


class RowRecorder:
    def __init__(self, csv_path=None):
        self._buf = io.StringIO()
        self._logger = rs.RunLogger(None)
        self._logger.f = self._buf           # RunLogger.log 는 self.f 에 한 줄 쓰고 flush 한다
        self._file = open(csv_path, "w", encoding="utf-8") if csv_path else None
        if self._file:
            self._file.write(rs.HEADER)

    def record(self, t, state, cmd) -> dict:
        self._buf.seek(0)
        self._buf.truncate()
        self._logger.log(t, state, cmd)
        line = self._buf.getvalue()
        if self._file:
            self._file.write(line)
        return parse_row(line)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
