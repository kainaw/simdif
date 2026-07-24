import pytest
from simdif.metrics.russel_rao import sim_russel_rao, dif_russel_rao
from simdif import russel_rao, sim, dif, simdif

def test_russel_rao():
    # Edge case: empty universe -> 1.0
    assert sim_russel_rao([], []) == 1.0
    # RR = n11 / N, N = n11+n10+n01+n00
    # No universe -> n00=0: n11=2, n10=1, n01=1 -> N=4 -> 2/4 = 0.5
    assert sim_russel_rao([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    # With n_universe=10 -> n00 = 10 - |A U B| = 10 - 4 = 6, N=10 -> 2/10 = 0.2
    assert sim_russel_rao([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(0.2)
    # dif = 1 - sim
    assert dif_russel_rao([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert dif_russel_rao([1, 2, 3], [1, 3, 5], n_universe=10) == pytest.approx(0.8)
    # Convenience name (default role sim)
    assert russel_rao([1, 2, 3], [1, 3, 5]) == pytest.approx(0.5)
    assert sim([1, 2, 3], [1, 3, 5], 'russel_rao') == pytest.approx(0.5)
    assert dif([1, 2, 3], [1, 3, 5], 'russel_rao') == pytest.approx(0.5)
    # simdif list/dict form with n_universe forwarded
    assert simdif([1, 2, 3], [1, 3, 5], ['russel_rao'], n_universe=10) == {
        'russel_rao': pytest.approx(0.2)
    }
