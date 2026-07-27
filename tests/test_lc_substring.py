import pytest
from simdif.metrics.lc_substring import (
    score_lc_substring, sim_lc_substring, dif_lc_substring,
    trace_lc_substring, matrix_lc_substring,
)
from simdif.metrics.lcs import score_lcs
from simdif import lc_substring, sim, dif, simdif, METRICS


def test_lc_substring_score():
    # Docstring example: the longest contiguous run shared by "ABABC" and
    # "BABCA" is "BABC".
    assert score_lc_substring("ABABC", "BABCA") == 4

    # Edge cases: nothing in common, and empty inputs.
    assert score_lc_substring("XYZ", "ABC") == 0
    assert score_lc_substring("", "") == 0
    assert score_lc_substring("", "abc") == 0

    # Identical strings: the whole string is the run.
    assert score_lc_substring("abc", "abc") == 3

    # Fully contained substring scores its own full length.
    assert score_lc_substring("BCD", "ABCDE") == 3

    # THE distinction from lc_subsequence: interleaved junk destroys
    # contiguity but not subsequence order.
    assert score_lc_substring("ABCDE", "AXBXCXDXE") == 1
    assert score_lcs("ABCDE", "AXBXCXDXE") == 5


def test_lc_substring_non_string_inputs():
    # Lists take the DP path rather than the difflib fast path; both must
    # agree. to_list("...") of a string yields characters, so the same run
    # is expected either way.
    assert score_lc_substring(list("ABABC"), list("BABCA")) == 4
    assert score_lc_substring([1, 2, 3, 4], [9, 2, 3, 9]) == 2
    assert trace_lc_substring([1, 2, 3, 4], [9, 2, 3, 9]) == [2, 3]

    # Word-level tokens, not characters.
    assert score_lc_substring(["the", "quick", "fox"], ["a", "quick", "fox"]) == 2


def test_lc_substring_sim_dif():
    # sim normalizes by the LONGER input (max), so a contained substring is
    # not 1.0 -- same convention as sim_prefix / sim_suffix.
    assert sim_lc_substring("BCD", "ABCDE") == pytest.approx(3/5)

    # 1.0 iff identical; 0.0 iff nothing shared.
    assert sim_lc_substring("abc", "abc") == 1.0
    assert dif_lc_substring("abc", "abc") == 0.0
    assert sim_lc_substring("XYZ", "ABC") == 0.0
    assert dif_lc_substring("XYZ", "ABC") == 1.0

    # Two empty inputs are identical.
    assert sim_lc_substring("", "") == 1.0

    # sim and dif are always complementary.
    for a, b in [("cat", "cot"), ("kitten", "sitting"), ("AB", "ABCDEFG"), ("", "xyz")]:
        assert sim_lc_substring(a, b) + dif_lc_substring(a, b) == pytest.approx(1.0)


def test_lc_substring_trace():
    assert trace_lc_substring("ABABC", "BABCA") == "BABC"
    assert trace_lc_substring("XYZ", "ABC") == ""
    assert trace_lc_substring("abc", "abc") == "abc"

    # Ties break toward the earliest occurrence in A: "AB" and "CD" both run
    # length 2, and "AB" comes first.
    assert trace_lc_substring("ABxCD", "ABzCD") == "AB"

    # The trace really is a contiguous slice of both inputs.
    a, b = "the quick brown fox", "a quick brown cat"
    run = trace_lc_substring(a, b)
    assert run in a and run in b
    assert len(run) == score_lc_substring(a, b)


def test_lc_substring_matrix():
    # A cell holds the run ending exactly there, so mismatches reset to 0 and
    # the grid is NOT monotone -- the score is the largest cell anywhere, not
    # the bottom-right corner.
    grid = matrix_lc_substring("ABABC", "BABCA")
    cells = [c for row in grid[1:] for c in row[1:] if isinstance(c, int)]
    assert max(cells) == 4
    assert grid[-1][-1] != 4


def test_lc_substring_registry_and_aliases():
    # Default role is 'score', and there is deliberately no 'dist' role.
    assert METRICS['lc_substring']['default'] == 'score'
    assert 'dist' not in METRICS['lc_substring']

    # Unambiguous aliases all resolve to the same metric.
    for alias in ['lcstr', 'lcsubstr', 'longest_common_substring']:
        assert METRICS[alias] is METRICS['lc_substring']

    # 'lcs' must NOT resolve here -- it stays the subsequence variant.
    assert METRICS['lcs'] is not METRICS['lc_substring']
    for alias in ['lc_subsequence', 'lcsubseq', 'longest_common_subsequence']:
        assert METRICS[alias] is METRICS['lcs']

    # Convenience/dispatcher access.
    assert lc_substring("ABABC", "BABCA") == 4
    assert sim("BCD", "ABCDE", "lc_substring") == pytest.approx(3/5)
    assert dif("BCD", "ABCDE", "lc_substring") == pytest.approx(2/5)
    assert simdif("ABABC", "BABCA", ["lc_substring"]) == {"lc_substring": 4}

    # The comparison a `skip=` flag could never have expressed: both variants
    # side by side in one call.
    assert simdif("ABCDE", "AXBXCXDXE", ["lc_subsequence", "lc_substring"]) == {
        "lc_subsequence": 5,
        "lc_substring": 1,
    }
