import pytest
from simdif.metrics.rand_index import sim_rand_index, dif_rand_index
from simdif.metrics._helpers import _pair_counts
from simdif import rand_index, sim, dif

# The lecture example: two clusterings of 5 objects.
C1 = "aabcc"   # {0,1}, {2}, {3,4}
C2 = "abbbc"   # {0}, {1,2,3}, {4}


def test_pair_counts():
    # a=together in both, b=together in A only, c=together in B only, d=apart in both.
    assert _pair_counts(list(C1), list(C2)) == (0, 2, 3, 5)


def test_rand_index():
    # RI = (a+d)/total = (0+5)/10 = 0.5
    assert sim_rand_index(C1, C2) == pytest.approx(0.5)
    assert dif_rand_index(C1, C2) == pytest.approx(0.5)  # (b+c)/total = 5/10
    # Identical clusterings -> 1.0.
    assert sim_rand_index(C1, C1) == pytest.approx(1.0)
    # Label-invariance: relabelling clusters leaves the same partition -> 1.0.
    assert sim_rand_index("aabcc", "bbcaa") == pytest.approx(1.0)
    # Length mismatch raises (can't compare clusterings of different object sets).
    with pytest.raises(ValueError, match="Vector length mismatch"):
        sim_rand_index("aab", "aabcc")
    # Dispatch + aliases.
    assert rand_index(C1, C2) == pytest.approx(0.5)
    assert sim(C1, C2, "rand") == pytest.approx(0.5)
    assert dif(C1, C2, "rand_index") == pytest.approx(0.5)
