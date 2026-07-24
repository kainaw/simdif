import pytest
from simdif.metrics.needleman_wunsch import score_needleman_wunsch, trace_needleman_wunsch
from simdif import needleman_wunsch, score, trace, simdif

def test_needleman_wunsch():
    # Edge case: two empty sequences align to a score of 0.
    assert score_needleman_wunsch("", "") == 0
    # Identical length-n sequences: n matches * match_score(1) = n.
    assert score_needleman_wunsch("AT", "AT") == 2
    # One match (+1) and one mismatch (-1) => 0.
    assert score_needleman_wunsch("AT", "AC") == 0
    # "AAT" vs "AT": best is 2 matches (+2) and 1 gap (-1) => 1.
    assert score_needleman_wunsch("AAT", "AT") == 1
    # Custom match_score kwarg: 2 matches * 2 = 4.
    assert score_needleman_wunsch("AT", "AT", match_score=2) == 4
    # Secondary role: trace returns the aligned pair.
    assert trace_needleman_wunsch("AT", "AT") == (["A", "T"], ["A", "T"])
    # Convenience name maps to the default 'score' role.
    assert needleman_wunsch("AT", "AT") == 2
    assert score("AT", "AT", "needleman_wunsch") == 2
    assert trace("AT", "AT", "needleman_wunsch") == (["A", "T"], ["A", "T"])
    assert simdif("AT", "AT", ["needleman_wunsch"]) == {"needleman_wunsch": 2}
