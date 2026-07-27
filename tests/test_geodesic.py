import math
import pytest
from simdif.metrics.geodesic import (dist_geodesic, dif_geodesic, sim_geodesic,
                                     dist_earth, dif_earth, sim_earth,
                                     EARTH_RADIUS_KM)
from simdif import geodesic, dist, dif, sim, simdif

def test_geodesic_basic():
    assert dist_geodesic([], [], radius=1) == pytest.approx(0.0)
    assert dist_geodesic([40.7128, 74.0060], [51.5074,-0.1278], radius=3959, unit='d') == pytest.approx(3471.819180509103)
    assert dist_geodesic([1], [1], radius=100) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_geodesic([1, 2, 3], [1, 2], radius=100)
    assert dist_geodesic([1], [2], radius=100) == pytest.approx(100.0)
    assert dist_geodesic([1,2], [1,2], radius=100) == pytest.approx(0.0)
    assert dist_geodesic([1,2], [2,3], radius=100) == pytest.approx(87.15212348676054)
    assert dist_geodesic([1,2,3], [1,2,3], radius=100) == pytest.approx(0.0)
    assert dist_geodesic([1,2,3], [2,3,4], radius=100) == pytest.approx(81.44811512244253)
    assert dist([1], [2], 'geodesic', radius=100) == pytest.approx(100.0)
    assert simdif([1], [2], ['geodesic'], radius=100) == {'geodesic': pytest.approx(100.0)}


def test_geodesic_bound_is_tight():
    # The central angle cannot exceed pi, so pi * radius is the real maximum
    # and antipodal points land exactly on dif = 1.0.
    assert dist_geodesic([0.0, 0.0], [0.0, math.pi]) == pytest.approx(math.pi)
    assert dif_geodesic([0.0, 0.0], [0.0, math.pi]) == pytest.approx(1.0)
    assert sim_geodesic([0.0, 0.0], [0.0, math.pi]) == pytest.approx(0.0)
    # A quarter turn is exactly half way to antipodal.
    assert dif_geodesic([0.0, 0.0], [0.0, math.pi / 2]) == pytest.approx(0.5)
    # Identical points.
    assert dif_geodesic([1, 2], [1, 2], radius=100) == pytest.approx(0.0)
    assert sim_geodesic([1, 2], [1, 2], radius=100) == pytest.approx(1.0)


def test_geodesic_dif_cancels_the_radius():
    # dif is central_angle / pi, so the radius divides back out: the same two
    # points score the same dif on any sphere, in any unit.
    a, b = [1, 2], [2, 3]
    base = dif_geodesic(a, b, radius=1)
    for radius in (100, 3959, EARTH_RADIUS_KM):
        assert dist_geodesic(a, b, radius=radius) != pytest.approx(dist_geodesic(a, b, radius=1))
        assert dif_geodesic(a, b, radius=radius) == pytest.approx(base)
        assert sim_geodesic(a, b, radius=radius) + dif_geodesic(a, b, radius=radius) == pytest.approx(1.0)


def test_earth_dif_is_unit_free():
    nyc, london = [40.7128, -74.0060], [51.5074, -0.1278]
    # Default radius is Earth's mean radius in km; the ceiling is the antipodal
    # distance, pi * 6371.0088 ~ 20015.09 km.
    assert dist_earth(nyc, london) == pytest.approx(5570.2299, rel=1e-6)
    km = dif_earth(nyc, london)
    miles = dif_earth(nyc, london, radius=3958.8)
    assert km == pytest.approx(miles)
    assert km == pytest.approx(5570.2299 / (math.pi * EARTH_RADIUS_KM), rel=1e-6)
    assert sim_earth(nyc, london) == pytest.approx(1.0 - km)
    # Antipodes of NYC: negate the latitude, shift longitude by 180.
    assert dif_earth(nyc, [-40.7128, 105.9940]) == pytest.approx(1.0)
    assert dif_earth(nyc, nyc) == pytest.approx(0.0)
    # Role dispatchers see the same values.
    assert dif(nyc, london, 'earth') == pytest.approx(km)
    assert sim(nyc, london, 'earth') == pytest.approx(1.0 - km)
