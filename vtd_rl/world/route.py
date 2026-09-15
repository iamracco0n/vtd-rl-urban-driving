"""경로 폴리라인 — 누적거리·투영·방위."""
import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    s: float          # 경로 누적거리[m]
    lateral: float    # 경로 진행 방향 왼쪽 + [m]
    index: int        # 가장 가까운 경로점 번호


class RouteIndex:
    def __init__(self, pts):
        if len(pts) < 2:
            raise ValueError("경로점이 2개 이상이어야 한다")
        self.pts = [(float(x), float(y)) for x, y in pts]
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.total = self.cum[-1]

    def heading_at(self, i: int) -> float:
        a = self.pts[max(0, i - 3)]
        b = self.pts[min(len(self.pts) - 1, i + 3)]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def project(self, x: float, y: float, hint=None, window: int = 60) -> "Projection":
        n = len(self.pts)
        rng = range(n) if hint is None else range(max(0, hint - window), min(n, hint + window + 1))
        i = min(rng, key=lambda j: (self.pts[j][0] - x) ** 2 + (self.pts[j][1] - y) ** 2)
        best = None
        for j in (i - 1, i):
            if j < 0 or j + 1 >= n:
                continue
            (x0, y0), (x1, y1) = self.pts[j], self.pts[j + 1]
            dx, dy = x1 - x0, y1 - y0
            seg2 = dx * dx + dy * dy
            u = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
            px, py = x0 + u * dx, y0 + u * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if best is None or d2 < best[0]:
                seg = math.sqrt(seg2)
                lat = 0.0 if seg == 0 else (dx * (y - y0) - dy * (x - x0)) / seg
                best = (d2, self.cum[j] + u * seg, lat)
        return Projection(best[1], best[2], i)

    def point_at(self, s: float):
        s = max(0.0, min(self.total, s))
        j = max(0, min(len(self.pts) - 2, bisect.bisect_right(self.cum, s) - 1))
        (x0, y0), (x1, y1) = self.pts[j], self.pts[j + 1]
        seg = self.cum[j + 1] - self.cum[j]
        u = 0.0 if seg == 0 else (s - self.cum[j]) / seg
        return x0 + u * (x1 - x0), y0 + u * (y1 - y0), math.atan2(y1 - y0, x1 - x0)
