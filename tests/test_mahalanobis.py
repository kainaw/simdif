import pytest
from simdif.metrics.mahalanobis import dist_mahalanobis, sim_mahalanobis
from simdif import mahalanobis, dist, simdif

def test_mahalanobis_basic():
    assert dist_mahalanobis([], []) == pytest.approx(0.0)
    # With no inverse-covariance supplied it reduces to Euclidean: sqrt(4^2 + 3^2) = 5
    assert dist_mahalanobis([0, 3], [4, 0]) == pytest.approx(5.0)
    # Identity inverse-covariance is exactly Euclidean. The kwarg name
    # 'covariance_inv' matches info_mahalanobis().
    assert dist_mahalanobis([0, 3], [4, 0], covariance_inv=[[1, 0], [0, 1]]) == pytest.approx(5.0)
    # diff = [-4, 3]; S_inv = 0.25*I -> sqrt(0.25 * (16 + 9)) = sqrt(6.25) = 2.5
    assert dist_mahalanobis([0, 3], [4, 0], covariance_inv=[[0.25, 0], [0, 0.25]]) == pytest.approx(2.5)
    # sim = 1 / (1 + d) = 1 / 6 for the Euclidean case
    assert sim_mahalanobis([0, 3], [4, 0]) == pytest.approx(1.0 / 6.0)
    # mahalanobis has no pad support; length mismatch raises the standard message.
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_mahalanobis([1, 2, 3], [1, 2])
    assert mahalanobis([0, 3], [4, 0]) == pytest.approx(5.0)
    assert dist([0, 3], [4, 0], 'mahalanobis') == pytest.approx(5.0)
    assert simdif([0, 3], [4, 0], ['mahalanobis']) == {'mahalanobis': pytest.approx(5.0)}
