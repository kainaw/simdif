import pytest
from simdif.metrics.canberra import dist_canberra, dif_canberra, sim_canberra
from simdif import canberra, dist, simdif

def test_canberra_basic():
    assert dist_canberra([], []) == pytest.approx(0.0)
    assert dist_canberra([1, 2], [1, 6]) == pytest.approx(0.5)
    assert dist_canberra([1, 2], [1], pad_value="6") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_canberra([1, 2, 3], [1, 2], pad_value=None)
    # dif = dist / n; n = 2 here, so 0.5 / 2 = 0.25, and sim = 1 - dif.
    assert dif_canberra([1, 2], [1, 6]) == pytest.approx(0.25)
    assert sim_canberra([1, 2], [1, 6]) == pytest.approx(0.75)
    assert canberra([1, 2], [1, 6]) == pytest.approx(0.5)
    assert dist([1, 2], [1, 6], 'canberra') == pytest.approx(0.5)
    assert simdif([1,2], [1,6], ['canberra']) == {'canberra': pytest.approx(0.5)}


def test_canberra_bound_is_tight():
    # Every term is |x-y|/(|x|+|y|) <= 1, and opposite signs attain exactly 1,
    # so n is a real maximum rather than a loose ceiling.
    assert dist_canberra([1, 2, 3], [-1, -2, -3]) == pytest.approx(3.0)
    assert dif_canberra([1, 2, 3], [-1, -2, -3]) == pytest.approx(1.0)
    assert sim_canberra([1, 2, 3], [-1, -2, -3]) == pytest.approx(0.0)
    # Identical vectors sit at the other end.
    assert dif_canberra([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert sim_canberra([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # Two empty vectors: n = 0, so dif would be 0/0. They are identical.
    assert dif_canberra([], []) == pytest.approx(0.0)
    assert sim_canberra([], []) == pytest.approx(1.0)


def test_canberra_sim_dif_complementary():
    for a, b in ([1, 2], [1, 6]), ([0, 5], [0, -5]), ([1, 1], [1, 1]), ([], []):
        assert sim_canberra(a, b) + dif_canberra(a, b) == pytest.approx(1.0)
