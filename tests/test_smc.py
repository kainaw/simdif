import pytest
from simdif.metrics.smc import sim_smc, dif_smc
from simdif import smc, sim, dif, simdif

def test_smc():
    # Edge case: empty universe -> 1.0
    assert sim_smc([], []) == 1.0
    # SMC = (n11 + n00) / N; no n_universe -> n00=0 (reduces to Jaccard)
    # n11=2, n10=1, n01=1, n00=0 -> N=4 -> 2/4 = 0.5
    assert sim_smc([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    # With n_universe=10 -> n00 = 10 - |A U B| = 10 - 4 = 6, N=10 -> (2+6)/10 = 0.8
    assert sim_smc([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(0.8)
    # dif = 1 - sim
    assert dif_smc([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert dif_smc([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(0.2)
    # Convenience name (default role sim)
    assert smc([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert sim([1, 2, 3], [1, 3, 5], 'smc') == pytest.approx(0.5)
    assert dif([1, 2, 3], [1, 3, 5], 'smc') == pytest.approx(0.5)
    # simdif list/dict form with n_universe forwarded
    assert simdif([1, 2, 3], [1, 3, 5], ['smc'], n_universe=10) == {
        'smc': pytest.approx(0.8)
    }
