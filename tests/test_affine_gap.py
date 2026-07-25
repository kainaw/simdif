import pytest
from simdif.metrics.affine_gap import score_affine_gap, trace_affine_gap
from simdif import affine_gap, score, trace, simdif

def test_affine_gap():
    # Edge case: two empty sequences align to a score of 0.
    assert score_affine_gap("", "") == 0
    # Identical length-n sequences: n matches * match_score(1) = n. No gaps
    # involved, so this should agree with plain Needleman-Wunsch.
    assert score_affine_gap("AT", "AT") == 2
    # One match (+1) and one mismatch (-1) => 0. Still no gaps.
    assert score_affine_gap("AT", "AC") == 0
    # Custom match_score kwarg: 2 matches * 2 = 4.
    assert score_affine_gap("AT", "AT", match_score=2) == 4

    # "AAT" vs "AT" needs a single gap. With the default gap_open=-10,
    # gap_extend=-1, that one gap position costs -10 (open, no extend needed
    # since the run is length 1): 2 matches (+2) - 1 mismatch-free gap (-10)
    # => -8. This is the case that actually distinguishes affine from linear
    # gap costs -- with linear gap_penalty=-1 (as in Needleman-Wunsch) the
    # same alignment scores 1, not -8.
    assert score_affine_gap("AAT", "AT") == -8
    # Same alignment, but with gap_open/gap_extend both set to -1 (i.e. no
    # extra cost for "opening" a gap) affine collapses to the linear-gap
    # case and matches plain NW's score_needleman_wunsch("AAT", "AT") == 1.
    assert score_affine_gap("AAT", "AT", gap_open=-1, gap_extend=-1) == 1

    # Secondary role: trace returns the aligned pair (gap on the shorter
    # side, marked with the default gap_symbol "-").
    assert trace_affine_gap("AT", "AT") == (["A", "T"], ["A", "T"])
    assert trace_affine_gap("AAT", "AT") == (["A", "A", "T"], ["-", "A", "T"])

    # local=True: no common symbols -> every running score clamps to 0.
    assert score_affine_gap("AAA", "TTT", local=True) == 0
    # local=True identical: 2 matches * match_score(2) = 4 (matches
    # score_smith_waterman("AT", "AT") == 4 when gaps aren't involved).
    assert score_affine_gap("AT", "AT", local=True, match_score=2) == 4
    # local=True single shared symbol 'A': one match * 2 = 2.
    assert score_affine_gap("XA", "YA", local=True, match_score=2) == 2
    assert trace_affine_gap("XA", "YA", local=True, match_score=2) == (["A"], ["A"])
    # local=True fully shared region: 3 matches * 2 = 6.
    assert score_affine_gap("CAB", "CAB", local=True, match_score=2) == 6

    # The actual point of affine gaps: one contiguous run of gaps should
    # score better than the same number of scattered single-position gaps
    # would under a linear-per-position cost. Here "AAAACCCCTTTTAAAA" has
    # one long non-matching run in the middle, and the local alignment
    # should grab the full contiguous "AAAA" match on either side rather
    # than fragmenting.
    assert score_affine_gap("AAAACCCCTTTTAAAA", "AAAAAAAA", local=True, match_score=2) == 8
    assert trace_affine_gap("AAAACCCCTTTTAAAA", "AAAAAAAA", local=True, match_score=2) == (
        ["A", "A", "A", "A"], ["A", "A", "A", "A"]
    )

    # Convenience name maps to the default 'score' role.
    assert affine_gap("AT", "AT") == 2
    assert score("AT", "AT", "affine_gap") == 2
    assert trace("AT", "AT", "affine_gap") == (["A", "T"], ["A", "T"])
    assert simdif("AT", "AT", ["affine_gap"]) == {"affine_gap": 2}
