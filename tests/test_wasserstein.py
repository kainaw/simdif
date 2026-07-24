import pytest
from simdif.metrics.wasserstein import dist_wasserstein
from simdif import wasserstein, dist, simdif

def test_wasserstein():
    # Identical distributions -> 0.
    assert dist_wasserstein([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # All mass moves 1 bin over: W_1 = 1.
    assert dist_wasserstein([1, 0], [0, 1]) == pytest.approx(1.0)
    # Mass moves 2 bins over: W_1 = 2.
    assert dist_wasserstein([1, 0, 0], [0, 0, 1]) == pytest.approx(2.0)
    # Known value (p=1): P=[0.5,0.5,0,0], Q=[0,0,0,1].
    #   t in (0,0.5]: width 0.5, |bin0 - bin3| = 3 -> 1.5
    #   t in (0.5,1]: width 0.5, |bin1 - bin3| = 2 -> 1.0
    #   W_1 = 2.5
    assert dist_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1]) == pytest.approx(2.5)
    # Order p changes the result: W_2 = (0.5*3^2 + 0.5*2^2)^(1/2) = sqrt(6.5).
    assert dist_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1], p=2) == pytest.approx(6.5 ** 0.5)
    # p must be positive.
    with pytest.raises(ValueError, match="must be positive"):
        dist_wasserstein([1, 0], [0, 1], p=0)
    # Convenience name (default role is 'dist') + role dispatcher + simdif dict form.
    assert wasserstein([1, 0, 0], [0, 0, 1]) == pytest.approx(2.0)
    assert dist([1, 0, 0], [0, 0, 1], 'wasserstein') == pytest.approx(2.0)
    assert simdif([1, 0, 0], [0, 0, 1], ['wasserstein']) == {'wasserstein': pytest.approx(2.0)}
    # Aliases earth_mover / emd resolve to the same metric.
    assert simdif([1, 0, 0], [0, 0, 1], ['emd']) == {'emd': pytest.approx(2.0)}
    assert dist([1, 0, 0], [0, 0, 1], 'earth_mover') == pytest.approx(2.0)
