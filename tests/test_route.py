import math

import pytest

from vtd_rl.world.route import RouteIndex


def straight():
    return RouteIndex([(float(i), 0.0) for i in range(21)])


def test_누적거리():
    r = straight()
    assert r.total == pytest.approx(20.0)
    assert r.cum[5] == pytest.approx(5.0)


def test_투영_왼쪽이_양수():
    r = straight()
    p = r.project(5.3, 2.0)
    assert p.s == pytest.approx(5.3)
    assert p.lateral == pytest.approx(2.0)
    q = r.project(15.0, -1.0)
    assert q.lateral == pytest.approx(-1.0)


def test_투영_힌트_창():
    r = straight()
    assert r.project(12.2, 0.5, hint=12).s == pytest.approx(12.2)


def test_경로_끝을_넘으면_끝에_붙는다():
    r = straight()
    p = r.project(25.0, 0.0)
    assert p.s == pytest.approx(20.0) and p.index == 20


def test_지점과_방위():
    r = RouteIndex([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    x, y, h = r.point_at(15.0)
    assert (x, y) == pytest.approx((10.0, 5.0))
    assert h == pytest.approx(math.pi / 2)
    assert r.heading_at(0) == pytest.approx(math.atan2(10.0, 10.0))


def test_점이_하나면_거부():
    with pytest.raises(ValueError):
        RouteIndex([(0.0, 0.0)])
