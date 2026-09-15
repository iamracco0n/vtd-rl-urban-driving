"""일치 검증 — 같은 행을 사후 채점기와 심판에 넣고 감점 호출을 비교한다."""
import collections
import time
from dataclasses import dataclass

from vtd_rl.referee.core import Referee
from vtd_rl.referee.oracle import score_episode

ALL_ITEMS = tuple(range(1, 16))


@dataclass
class Parity:
    want: collections.Counter
    got: collections.Counter
    respawns_want: dict
    respawns_got: dict
    state_want: object          # 전 항목 비교일 때만 list, 아니면 None
    state_got: object
    frames: int
    referee_seconds: float

    @property
    def ok(self) -> bool:
        return (self.want == self.got and self.respawns_want == self.respawns_got
                and self.state_want == self.state_got)

    @property
    def per_frame_us(self) -> float:
        return self.referee_seconds / max(self.frames, 1) * 1e6

    def diff(self) -> str:
        return (f"채점기에만 {dict(self.want - self.got)} / 심판에만 {dict(self.got - self.want)} / "
                f"리스폰 채점기 {self.respawns_want} 심판 {self.respawns_got} / "
                f"Sheet 채점기 {self.state_want} 심판 {self.state_got}")


def compare(board, rows, items=ALL_ITEMS, sections=5, use_map=True) -> Parity:
    items = tuple(items)
    oracle = score_episode(rows, board, sections, use_map)
    ref = Referee(board, sections, use_map)
    t0 = time.perf_counter()
    for r in rows:
        ref.step(dict(r))
    ref.finish()
    seconds = time.perf_counter() - t0
    want = collections.Counter(h for h in oracle.hits if h[1] in items)
    got = collections.Counter((h.sec, h.item, h.level) for h in ref.hits
                              if h.item in items and h.item != 15)
    full = items == ALL_ITEMS
    return Parity(want, got,
                  oracle.respawns if 15 in items else {}, ref.respawns if 15 in items else {},
                  oracle.state if full else None,
                  [dict(s) for s in ref.sheet.state] if full else None,
                  len(rows), seconds)
