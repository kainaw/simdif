import pytest
from simdif.metrics.sokal_sneath_i import sim_sokal_sneath_i, dif_sokal_sneath_i
from simdif import sokal_sneath_i, sim, dif, simdif

def test_sokal_sneath_i():
    # Edge case: empty/empty -> denominator 0 -> 1.0
    assert sim_sokal_sneath_i([], []) == 1.0
    # SSI = n11 / (n11 + 2*(n10+n01))
    # n11=2, n10=1, n01=1 -> 2 / (2 + 2*2) = 2/6 = 1/3
    assert sim_sokal_sneath_i([1, 2, 3], [1, 3, 5]) == pytest.approx(1/3)
    # Identical sets: n11=2, mismatches=0 -> 2/2 = 1.0
    assert sim_sokal_sneath_i([1, 2], [1, 2]) == pytest.approx(1.0)
    # dif = 1 - sim -> 1 - 1/3 = 2/3
    assert dif_sokal_sneath_i([1, 2, 3], [1, 3, 5]) == pytest.approx(2/3)
    # Convenience name (default role sim)
    assert sokal_sneath_i([1, 2, 3], [1, 3, 5]) == pytest.approx(1/3)
    assert sim([1, 2, 3], [1, 3, 5], 'sokal_sneath_i') == pytest.approx(1/3)
    assert dif([1, 2, 3], [1, 3, 5], 'sokal_sneath_i') == pytest.approx(2/3)
    # simdif list/dict form
    assert simdif([1, 2, 3], [1, 3, 5], ['sokal_sneath_i']) == {
        'sokal_sneath_i': pytest.approx(1/3)
    }
