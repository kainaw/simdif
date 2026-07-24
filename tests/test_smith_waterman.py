import pytest
from simdif.metrics.smith_waterman import score_smith_waterman, trace_smith_waterman
from simdif import smith_waterman, score, trace, simdif

def test_smith_waterman():
    # No common symbols: every running score clamps to 0.
    assert score_smith_waterman("AAA", "TTT") == 0
    # Identical: 2 matches * match_score(2) = 4.
    assert score_smith_waterman("AT", "AT") == 4
    # Single shared symbol 'A': one match * 2 = 2.
    assert score_smith_waterman("XA", "YA") == 2
    # Fully shared local region: 3 matches * 2 = 6.
    assert score_smith_waterman("CAB", "CAB") == 6
    # Secondary role: trace returns just the best local region.
    assert trace_smith_waterman("XA", "YA") == (["A"], ["A"])
    # Convenience name maps to the default 'score' role.
    assert smith_waterman("AT", "AT") == 4
    assert score("AT", "AT", "smith_waterman") == 4
    assert trace("XA", "YA", "smith_waterman") == (["A"], ["A"])
    assert simdif("AT", "AT", ["smith_waterman"]) == {"smith_waterman": 4}
