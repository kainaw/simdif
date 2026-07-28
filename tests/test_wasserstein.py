import pytest
from simdif.metrics.wasserstein import dist_wasserstein, sim_wasserstein, dif_wasserstein, explain_wasserstein
from simdif import wasserstein, dist, sim, dif, simdif

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


def test_wasserstein_sim_dif():
    # Identical distributions -> sim 1, dif 0.
    assert sim_wasserstein([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    assert dif_wasserstein([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # Default squash: sim = 1/(1+D), dif = 1 - sim.
    assert sim_wasserstein([1, 0], [0, 1]) == pytest.approx(1 / (1 + 1.0))
    assert dif_wasserstein([1, 0], [0, 1]) == pytest.approx(1 - 1 / (1 + 1.0))
    assert sim_wasserstein([1, 0], [0, 1]) + dif_wasserstein([1, 0], [0, 1]) == pytest.approx(1.0)
    # d_max makes dif the linear rescale D / d_max, sim = 1 - dif.
    assert dif_wasserstein([1, 0], [0, 1], d_max=2) == pytest.approx(0.5)
    assert sim_wasserstein([1, 0], [0, 1], d_max=2) == pytest.approx(0.5)
    # Values beyond d_max clamp to dif=1.0 / sim=0.0.
    assert dif_wasserstein([1, 0, 0], [0, 0, 1], d_max=1) == pytest.approx(1.0)
    assert sim_wasserstein([1, 0, 0], [0, 0, 1], d_max=1) == pytest.approx(0.0)
    # Role dispatcher + convenience names.
    assert sim([1, 0], [0, 1], 'wasserstein') == pytest.approx(1 / (1 + 1.0))
    assert dif([1, 0], [0, 1], 'emd') == pytest.approx(1 - 1 / (1 + 1.0))


def test_wasserstein_optimized_lib_scipy(optimized_lib):
    # scipy's wasserstein_distance path only engages for p=1.
    optimized_lib('scipy')
    assert dist_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1]) == pytest.approx(2.5)
    assert "Note:" not in explain_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1])


def test_wasserstein_optimized_lib_ot(optimized_lib):
    # ot (POT)'s wasserstein_1d path is used regardless of p.
    optimized_lib('ot')
    assert dist_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1]) == pytest.approx(2.5)
    assert dist_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1], p=2) == pytest.approx(6.5 ** 0.5)
    assert "Note:" not in explain_wasserstein([0.5, 0.5, 0, 0], [0, 0, 0, 1])
