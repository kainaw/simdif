import math
import pytest
from simdif.metrics.kimura import (
    dist_kimura, sim_kimura, _transitions_transversions, _DNA_GROUPS,
)
from simdif import kimura, simdif


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
