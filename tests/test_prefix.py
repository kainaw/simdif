import pytest
from simdif.metrics.prefix import score_prefix, sim_prefix, dif_prefix
from simdif import prefix, score, sim, dif, simdif

def test_prefix():
    # Edge case: two empty sequences share a trivial (empty) prefix, and
    # are treated as identical -> sim=1, dif=0.
    assert score_prefix("", "") == 0
    assert sim_prefix("", "") == 1.0
    assert dif_prefix("", "") == 0.0

    # Identical strings: whole thing is the common prefix.
    assert score_prefix("abc", "abc") == 3
    assert sim_prefix("abc", "abc") == 1.0

    # "prefix"/"preheat": share "pre" (3), then 'f' != 'h'.
    assert score_prefix("prefix", "preheat") == 3
    assert sim_prefix("prefix", "preheat") == pytest.approx(3/7)
    assert dif_prefix("prefix", "preheat") == pytest.approx(4/7)

    # No common prefix at all despite sharing a common SUFFIX -- prefix
    # similarity is blind to anything but the start of the sequence.
    assert score_prefix("suffix", "postfix") == 0
    assert sim_prefix("suffix", "postfix") == 0.0

    # No overlap whatsoever.
    assert score_prefix("abc", "xyz") == 0
    assert sim_prefix("abc", "xyz") == 0.0

    # One string is a strict prefix of the other -- shorter string is
    # fully "used up", but sim is normalized by the LONGER string's
    # length, so it's not 1.0 (mirrors sim_levenshtein's convention).
    assert score_prefix("test", "testing") == 4
    assert sim_prefix("test", "testing") == pytest.approx(4/7)

    # sim and dif are always complementary.
    for a, b in [("interstate", "internet"), ("a", "abcdefgh"), ("", "xyz")]:
        assert sim_prefix(a, b) + dif_prefix(a, b) == pytest.approx(1.0)

    # Convenience name (default role is 'score').
    assert prefix("prefix", "preheat") == 3
    assert score("prefix", "preheat", "prefix") == 3
    assert sim("prefix", "preheat", "prefix") == pytest.approx(3/7)
    assert dif("prefix", "preheat", "prefix") == pytest.approx(4/7)
    assert simdif("prefix", "preheat", ["prefix"]) == {"prefix": 3}
