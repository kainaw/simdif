import pytest
from simdif.metrics.spearman import sim_spearman, dist_spearman
from simdif import spearman, sim, simdif

def test_spearman():
    # Perfect positive monotonic relationship.
    assert sim_spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    # Perfect negative monotonic relationship.
    assert sim_spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Monotonic but non-linear (squares) => ranks identical => 1.0.
    assert sim_spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    # Hand-computed: ranks a=[1,2,3], b=[1,3,2].
    # centered a=[-1,0,1], b=[-1,1,0]; num=1, denom=sqrt2*sqrt2=2 => 0.5.
    assert sim_spearman([1, 2, 3], [1, 3, 2]) == pytest.approx(0.5)
    # sim + dist == 1.
    assert sim_spearman([1, 2, 3], [1, 3, 2]) + dist_spearman([1, 2, 3], [1, 3, 2]) == pytest.approx(1.0)
    # Error handling.
    with pytest.raises(ValueError, match="at least 2 elements"):
        sim_spearman([1], [1])
    with pytest.raises(ValueError, match="same length"):
        sim_spearman([1, 2, 3], [1, 2])
    # Convenience name (default 'sim' role) and dispatchers.
    assert spearman([1, 2, 3], [1, 3, 2]) == pytest.approx(0.5)
    assert sim([1, 2, 3], [1, 3, 2], "spearman") == pytest.approx(0.5)
    assert simdif([1, 2, 3, 4], [1, 2, 3, 4], ["spearman"]) == {
        "spearman": pytest.approx(1.0)
    }
