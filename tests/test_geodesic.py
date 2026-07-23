import pytest
from simdif.metrics.geodesic import dist_geodesic
from simdif import geodesic, dist, simdif

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
