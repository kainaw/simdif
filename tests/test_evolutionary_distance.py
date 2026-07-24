import math
import pytest
from simdif.metrics.evolutionary_distance import (
    dist_p_distance, sim_p_distance,
    dist_jukes_cantor, sim_jukes_cantor,
    dist_kimura, sim_kimura,
    _transitions_transversions, _DNA_GROUPS,
)
from simdif import p_distance, jukes_cantor, kimura, dist, sim, simdif


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


def test_jukes_cantor():
    assert dist_jukes_cantor("ATGC", "ATGC") == pytest.approx(0.0)
    # p = 0.5, k = 4: d = -(3/4) ln(1 - (4/3)*0.5)
    expected = -(3 / 4) * math.log(1 - (4 / 3) * 0.5)
    assert dist_jukes_cantor("AAAA", "AAGG") == pytest.approx(expected)
    assert sim_jukes_cantor("AAAA", "AAGG") == pytest.approx(1 / (1 + expected))
    # k is a parameter: same p=0.25, different k -> different distance.
    p = 0.25  # "AAAA" vs "AAAG"
    assert dist_jukes_cantor("AAAA", "AAAG", k=4) == pytest.approx(-(3 / 4) * math.log(1 - (4 / 3) * p))
    assert dist_jukes_cantor("AAAA", "AAAG", k=2) == pytest.approx(-(1 / 2) * math.log(1 - 2 * p))
    # Saturation: p = 1.0 (>= 3/4) -> inf.
    assert math.isinf(dist_jukes_cantor("AAAA", "GGGG"))
    assert sim_jukes_cantor("AAAA", "GGGG") == pytest.approx(0.0)
    # Dispatch + alias.
    assert jukes_cantor("AAAA", "AAGG") == pytest.approx(expected)
    assert simdif("AAAA", "AAGG", ['jc69']) == {'jc69': pytest.approx(expected)}


def test_kimura():
    assert dist_kimura("ATGC", "ATGC") == pytest.approx(0.0)
    # "AAAAAAAA" vs "GAAAAAAC": 1 transition (A/G), 1 transversion (A/C), 6 matches.
    assert _transitions_transversions("AAAAAAAA", "GAAAAAAC", _DNA_GROUPS) == (8, 1, 1)
    #   P = Q = 1/8; d = -1/2 ln(1-2P-Q) - 1/4 ln(1-2Q)
    expected = -0.5 * math.log(1 - 2 * (1 / 8) - (1 / 8)) - 0.25 * math.log(1 - 2 * (1 / 8))
    assert dist_kimura("AAAAAAAA", "GAAAAAAC") == pytest.approx(expected)
    assert sim_kimura("AAAAAAAA", "GAAAAAAC") == pytest.approx(1 / (1 + expected))
    # Pairwise deletion: 'N' is outside every group -> that site is skipped.
    assert _transitions_transversions("AANA", "GAAA", _DNA_GROUPS) == (3, 1, 0)
    assert dist_kimura("AANA", "GAAA") == pytest.approx(-0.5 * math.log(1 / 3))
    # Saturation: too many transversions -> inf ("AAAA" vs "GCTA": P=1/4, Q=2/4).
    assert math.isinf(dist_kimura("AAAA", "GCTA"))
    # Custom groups make it generic (a 2-symbol partition of arbitrary tokens).
    ti_tv = _transitions_transversions(
        ['x', 'x'], ['y', 'z'], [frozenset({'x', 'y'}), frozenset({'z'})])
    assert ti_tv == (2, 1, 1)  # x/y same group -> transition; x/z -> transversion
    # Dispatch + alias.
    assert kimura("AAAAAAAA", "GAAAAAAC") == pytest.approx(expected)
    assert simdif("AAAAAAAA", "GAAAAAAC", ['k80']) == {'k80': pytest.approx(expected)}
