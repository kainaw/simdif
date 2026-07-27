import math
import pytest
from simdif.metrics.geodesic import (dist_geodesic, dif_geodesic, sim_geodesic,
                                     explain_geodesic, dist_earth, dif_earth,
                                     sim_earth, explain_earth, EARTH_RADIUS_KM)
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


def test_geodesic_length_mismatch_raises_from_both_roles():
    # geodesic uses to_list_numeric_aligned like the other vector metrics, so a
    # mismatch raises from explain_ too. It used to RETURN "Error: Vector length
    # mismatch" as a string, which a caller would print as if it were output.
    for fn in (dist_geodesic, dif_geodesic, sim_geodesic, explain_geodesic):
        with pytest.raises(ValueError, match="Vector length mismatch"):
            fn([1, 2, 3], [1, 2])
    for fn in (dist_earth, dif_earth, sim_earth, explain_earth):
        with pytest.raises(ValueError, match="Vector length mismatch"):
            fn([1, 2, 3], [1, 2])


def test_pad_value_zero_is_isometric():
    # The length of an angle list is the manifold dimension, so padding lifts
    # the shorter point onto the larger sphere. pad_value=0 appends a zero
    # angle, which inserts a zero Cartesian coordinate -- an embedding onto a
    # subsphere that moves nothing, so every distance survives the lift.
    nyc, lond = [40.7128, -74.0060], [51.5074, -0.1278]
    base = dist_earth(nyc, lond)
    kw = dict(radius=EARTH_RADIUS_KM, unit='degrees')
    # Both sides lifted explicitly.
    assert dist_geodesic(nyc + [0.0], lond + [0.0], **kw) == pytest.approx(base)
    # One side lifted by pad_value.
    assert dist_geodesic(nyc, lond + [0.0], pad_value=0, **kw) == pytest.approx(base)
    assert dist_geodesic(nyc + [0.0], lond, pad_value=0, **kw) == pytest.approx(base)
    # dif is unaffected too, since the manifold's pi*radius ceiling is unchanged.
    assert dif_geodesic(nyc, lond + [0.0], pad_value=0, **kw) == pytest.approx(dif_earth(nyc, lond))


def test_nonzero_pad_value_tilts_off_the_subsphere():
    # Documented as a caller's choice rather than a lift: any pad_value but 0
    # moves the padded point, monotonically further as the angle grows.
    nyc, lond3 = [40.7128, -74.0060], [51.5074, -0.1278, 0.0]
    kw = dict(radius=EARTH_RADIUS_KM, unit='degrees')
    d0 = dist_geodesic(nyc, lond3, pad_value=0, **kw)
    d30 = dist_geodesic(nyc, lond3, pad_value=30, **kw)
    d90 = dist_geodesic(nyc, lond3, pad_value=90, **kw)
    assert d0 < d30 < d90
    assert d0 == pytest.approx(dist_earth(nyc, [51.5074, -0.1278]))


def test_pad_value_is_in_the_input_unit():
    # Padding happens inside the coercer, before the degrees-to-radians
    # conversion, so pad_value is in the same unit as A and B -- not always
    # radians. 90 degrees and pi/2 radians must agree.
    degrees = dist_geodesic([0, 0], [0, 0, 0], unit='degrees', pad_value=90)
    radians = dist_geodesic([0, 0], [0, 0, 0], unit='radians', pad_value=math.pi / 2)
    assert degrees == pytest.approx(radians)
    assert degrees == pytest.approx(math.pi / 2)


def test_earth_validates_after_aligning():
    # pad_value can lift a lone latitude into a [lat, lon] pair...
    assert dist_earth([40.7], [51.5, -0.1], pad_value=0) == pytest.approx(1200.9313, rel=1e-6)
    # ...but no pad_value makes a 3-angle input a lat/lon pair, so the earth
    # check still fires on equal-but-wrong lengths.
    with pytest.raises(ValueError, match="requires \\[latitude, longitude\\] pairs"):
        dist_earth([1, 2, 3], [4, 5, 6])
    with pytest.raises(ValueError, match="requires \\[latitude, longitude\\] pairs"):
        dist_earth([1, 2, 3], [4, 5], pad_value=0)


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
