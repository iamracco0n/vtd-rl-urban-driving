"""score_fma.Sections 와 같은 답을 빠르게 — 경로점 격자 색인.

원본 index_of 는 경로점 전체를 훑어 가장 가까운 점 번호를 준다(거리가 같으면 앞 번호).
여기서는 CELL 격자의 고리를 안에서 밖으로 넓혀 간다. 고리 r 까지 본 뒤 남은 점은 모두
r*CELL + (자기 칸 경계까지 거리) 이상 떨어져 있다. 지금까지의 최소 거리가 그보다 작으면 멈춘다.
거리 식은 원본과 같은 식으로 계산해 동률 판단도 원본과 같게 한다.
"""
import math

from vtd_rl import rule_stack as rs

CELL = 12.0


class FastSections(rs.score_fma.Sections):
    def __init__(self, route, n):
        super().__init__(route, n)
        self._grid = {}
        for i, (x, y) in enumerate(self.route):
            self._grid.setdefault((math.floor(x / CELL), math.floor(y / CELL)), []).append(i)
        self._last = (None, None, None)       # 같은 행을 여러 판정기가 물으므로 직전 답을 둔다

    def index_of(self, x, y):
        if self._last[0] == x and self._last[1] == y:
            return self._last[2]
        cx, cy = math.floor(x / CELL), math.floor(y / CELL)
        edge = min(x - cx * CELL, (cx + 1) * CELL - x, y - cy * CELL, (cy + 1) * CELL - y)
        best_i, best_d = None, None
        ring = 0
        while True:
            for gx in range(cx - ring, cx + ring + 1):
                for gy in range(cy - ring, cy + ring + 1):
                    if max(abs(gx - cx), abs(gy - cy)) != ring:
                        continue
                    for i in self._grid.get((gx, gy), ()):
                        d = (self.route[i][0] - x) ** 2 + (self.route[i][1] - y) ** 2
                        if best_d is None or d < best_d or (d == best_d and i < best_i):
                            best_i, best_d = i, d
            bound = ring * CELL + edge
            if best_d is not None and best_d < bound * bound:
                break
            ring += 1
        self._last = (x, y, best_i)
        return best_i
