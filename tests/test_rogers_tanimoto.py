import pytest
from simdif.metrics.rogers_tanimoto import sim_rogers_tanimoto, dif_rogers_tanimoto
from simdif import rogers_tanimoto, simdif

def test_rogers_tanimoto():
    # Empty sets: denominator == 0 -> defined as identical
    assert sim_rogers_tanimoto([], []) == 1.0
    # A={1,2,3}, B={1,3,5}, n_universe=10 -> a=2,b=1,c=1,d=6
    # RT = (a+d) / (a + 2(b+c) + d) = (2+6)/(2+2*2+6) = 8/12 = 2/3
    assert sim_rogers_tanimoto([1,2,3],[1,3,5], n_universe=10) == pytest.approx(2/3)
    # Identical sets with a universe -> 1.0
    assert sim_rogers_tanimoto([1,2,3],[1,2,3], n_universe=10) == pytest.approx(1.0)
    # dif = 1 - sim = 1/3
    assert dif_rogers_tanimoto([1,2,3],[1,3,5], n_universe=10) == pytest.approx(1/3)
    # convenience name resolves to the default (sim) role
    assert rogers_tanimoto([1,2,3],[1,3,5], n_universe=10) == pytest.approx(2/3)
    assert simdif([1,2,3],[1,3,5],['rogers_tanimoto'], n_universe=10) == {
        'rogers_tanimoto': pytest.approx(2/3)
    }
