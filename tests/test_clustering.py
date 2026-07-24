import math
import pytest
from simdif.metrics.clustering import (
    sim_rand_index, dif_rand_index, sim_adjusted_rand, sim_fowlkes_mallows,
    _pair_counts,
)
from simdif import rand_index, adjusted_rand, fowlkes_mallows, sim, dif, simdif

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


def test_adjusted_rand():
    # ARI = 2(ad-bc)/((a+b)(b+d)+(a+c)(c+d)) = 2(0-6)/(2*7 + 3*8) = -12/38
    assert sim_adjusted_rand(C1, C2) == pytest.approx(-12 / 38)
    # Raw Rand looked "middling" (0.5) but ARI reveals worse-than-chance.
    assert sim_adjusted_rand(C1, C2) < 0
    # Identical -> 1.0; label-invariant.
    assert sim_adjusted_rand(C1, C1) == pytest.approx(1.0)
    assert sim_adjusted_rand("aabcc", "bbcaa") == pytest.approx(1.0)


def test_fowlkes_mallows():
    # FM = a/sqrt((a+b)(a+c)) = 0/sqrt(2*3) = 0.0 (no pair together in both).
    assert sim_fowlkes_mallows(C1, C2) == pytest.approx(0.0)
    # Identical -> 1.0.
    assert sim_fowlkes_mallows(C1, C1) == pytest.approx(1.0)
    # A clean partial case: FM = a/sqrt((a+b)(a+c)).
    a, b, c, d = _pair_counts(list("aaabb"), list("aabbb"))
    assert sim_fowlkes_mallows("aaabb", "aabbb") == pytest.approx(a / math.sqrt((a + b) * (a + c)))


def test_dispatch_and_aliases():
    assert rand_index(C1, C2) == pytest.approx(0.5)             # convenience name
    assert sim(C1, C2, "rand") == pytest.approx(0.5)            # alias
    assert dif(C1, C2, "rand_index") == pytest.approx(0.5)
    assert adjusted_rand(C1, C2) == pytest.approx(-12 / 38)
    assert fowlkes_mallows(C1, C2) == pytest.approx(0.0)
    for name in ("ari", "adjusted_rand_index"):
        assert simdif(C1, C2, [name]) == {name: pytest.approx(-12 / 38)}
    assert simdif(C1, C2, ["fm"]) == {"fm": pytest.approx(0.0)}
