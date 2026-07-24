import pytest
from simdif.metrics.p_distance import dist_p_distance, sim_p_distance
from simdif import p_distance, dist, simdif


def test_p_distance():
    assert dist_p_distance("ATGC", "ATGC") == pytest.approx(0.0)   # identical
    # "AAAA" vs "AAGG": 2 of 4 sites differ -> p = 0.5
    assert dist_p_distance("AAAA", "AAGG") == pytest.approx(0.5)
    assert sim_p_distance("AAAA", "AAGG") == pytest.approx(0.5)     # proportion identical
    # Generic: works on non-DNA tokens (== only).
    assert dist_p_distance(['x', 'y', 'z'], ['x', 'q', 'z']) == pytest.approx(1 / 3)
    # Length mismatch raises like other sequence metrics; pad_value is honored.
    with pytest.raises(ValueError, match="Vector length mismatch"):
        dist_p_distance("AAA", "AA")
    assert dist_p_distance("AAA", "AA", pad_value="A") == pytest.approx(0.0)
    # Dispatch forms.
    assert p_distance("AAAA", "AAGG") == pytest.approx(0.5)
    assert dist("AAAA", "AAGG", 'p_distance') == pytest.approx(0.5)
    assert simdif("AAAA", "AAGG", ['p_distance']) == {'p_distance': pytest.approx(0.5)}
