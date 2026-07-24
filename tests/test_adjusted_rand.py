import pytest
from simdif.metrics.adjusted_rand import sim_adjusted_rand
from simdif import adjusted_rand, simdif

# The lecture example: two clusterings of 5 objects.
C1 = "aabcc"   # {0,1}, {2}, {3,4}
C2 = "abbbc"   # {0}, {1,2,3}, {4}


def test_adjusted_rand():
    # ARI = 2(ad-bc)/((a+b)(b+d)+(a+c)(c+d)) = 2(0-6)/(2*7 + 3*8) = -12/38
    assert sim_adjusted_rand(C1, C2) == pytest.approx(-12 / 38)
    # Raw Rand looked "middling" (0.5) but ARI reveals worse-than-chance.
    assert sim_adjusted_rand(C1, C2) < 0
    # Identical -> 1.0; label-invariant.
    assert sim_adjusted_rand(C1, C1) == pytest.approx(1.0)
    assert sim_adjusted_rand("aabcc", "bbcaa") == pytest.approx(1.0)
    # Dispatch + aliases.
    assert adjusted_rand(C1, C2) == pytest.approx(-12 / 38)
    for name in ("ari", "adjusted_rand_index"):
        assert simdif(C1, C2, [name]) == {name: pytest.approx(-12 / 38)}
