import pytest
from simdif.metrics.suffix import score_suffix, sim_suffix, dif_suffix
from simdif import suffix, score, sim, dif, simdif

def test_suffix():
    # Edge case: two empty sequences share a trivial (empty) suffix, and
    # are treated as identical -> sim=1, dif=0.
    assert score_suffix("", "") == 0
    assert sim_suffix("", "") == 1.0
    assert dif_suffix("", "") == 0.0

    # Identical strings: whole thing is the common suffix.
    assert score_suffix("abc", "abc") == 3
    assert sim_suffix("abc", "abc") == 1.0

    # "suffix"/"postfix": share "fix" (3), counting backward from the end.
    assert score_suffix("suffix", "postfix") == 3
    assert sim_suffix("suffix", "postfix") == pytest.approx(3/7)
    assert dif_suffix("suffix", "postfix") == pytest.approx(4/7)

    # No common suffix at all despite sharing a common PREFIX -- suffix
    # similarity is blind to anything but the end of the sequence.
    assert score_suffix("prefix", "preheat") == 0
    assert sim_suffix("prefix", "preheat") == 0.0

    # No overlap whatsoever.
    assert score_suffix("abc", "xyz") == 0
    assert sim_suffix("abc", "xyz") == 0.0

    # "test"/"testing" share a common PREFIX but no common suffix
    # ('t' vs 'g' at the end) -- contrast with test_prefix's version of
    # this same pair.
    assert score_suffix("test", "testing") == 0
    assert sim_suffix("test", "testing") == 0.0

    # sim and dif are always complementary.
    for a, b in [("interstate", "internet"), ("a", "abcdefgh"), ("", "xyz")]:
        assert sim_suffix(a, b) + dif_suffix(a, b) == pytest.approx(1.0)

    # Convenience name (default role is 'score').
    assert suffix("suffix", "postfix") == 3
    assert score("suffix", "postfix", "suffix") == 3
    assert sim("suffix", "postfix", "suffix") == pytest.approx(3/7)
    assert dif("suffix", "postfix", "suffix") == pytest.approx(4/7)
    assert simdif("suffix", "postfix", ["suffix"]) == {"suffix": 3}
