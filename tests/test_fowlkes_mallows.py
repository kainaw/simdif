import math
import pytest
from simdif.metrics.fowlkes_mallows import sim_fowlkes_mallows
from simdif.metrics._helpers import _pair_counts
from simdif import fowlkes_mallows, simdif

# The lecture example: two clusterings of 5 objects.
C1 = "aabcc"   # {0,1}, {2}, {3,4}
C2 = "abbbc"   # {0}, {1,2,3}, {4}


def test_fowlkes_mallows():
    # FM = a/sqrt((a+b)(a+c)) = 0/sqrt(2*3) = 0.0 (no pair together in both).
    assert sim_fowlkes_mallows(C1, C2) == pytest.approx(0.0)
    # Identical -> 1.0.
    assert sim_fowlkes_mallows(C1, C1) == pytest.approx(1.0)
    # A clean partial case: FM = a/sqrt((a+b)(a+c)).
    a, b, c, d = _pair_counts(list("aaabb"), list("aabbb"))
    assert sim_fowlkes_mallows("aaabb", "aabbb") == pytest.approx(a / math.sqrt((a + b) * (a + c)))
    # Dispatch + aliases.
    assert fowlkes_mallows(C1, C2) == pytest.approx(0.0)
    assert simdif(C1, C2, ["fm"]) == {"fm": pytest.approx(0.0)}
