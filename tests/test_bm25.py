import math
import pytest
from simdif.metrics.bm25 import score_bm25
from simdif import bm25, score, simdif


def test_bm25_basic():
    query = ['cat']
    document = ['the', 'cat', 'sat', 'on', 'the', 'cat']  # tf('cat')=2, |D|=6
    corpus = [
        ['the', 'cat', 'sat'],          # contains cat
        ['the', 'dog', 'ran'],          # no cat
        ['a', 'cat', 'and', 'a', 'dog'],  # contains cat
        ['the', 'the', 'the'],          # no cat
    ]
    # N=4, df('cat')=2, avgdl=(3+3+5+3)/4=3.5, |D|=6, k1=1.5, b=0.75
    # idf = ln((4-2+0.5)/(2+0.5) + 1) = ln(2) = 0.693147...
    # denom = 2 + 1.5*(1 - 0.75 + 0.75*6/3.5) = 4.303571...
    # score = idf * (2*2.5)/denom = 0.693147 * 5 / 4.303571 = 0.805292...
    idf = math.log(2.0)
    denom = 2 + 1.5 * (1 - 0.75 + 0.75 * 6 / 3.5)
    expected = idf * (2 * 2.5) / denom
    assert score_bm25(query, document, corpus=corpus) == pytest.approx(expected)

    # A query term absent from the document contributes nothing.
    assert score_bm25(['zebra'], document, corpus=corpus) == pytest.approx(0.0)
    # Empty query -> zero score.
    assert score_bm25([], document, corpus=corpus) == pytest.approx(0.0)

    # k1 / b_norm are tunable.
    denom0 = 2 + 0.0 * (1 - 0.75 + 0.75 * 6 / 3.5)  # k1=0 -> denom = tf
    assert score_bm25(query, document, corpus=corpus, k1=0.0) == pytest.approx(
        idf * (2 * 1.0) / denom0)

    # Missing corpus falls back to [query, document] (2-doc corpus), and must
    # equal passing that same corpus explicitly.
    default_score = score_bm25(query, document)
    explicit = score_bm25(query, document, corpus=[query, document])
    assert default_score == pytest.approx(explicit)

    # Convenience name (default role 'score'), score() dispatcher, simdif dict,
    # and the 'okapi' alias all resolve and forward the corpus kwarg.
    assert bm25(query, document, corpus=corpus) == pytest.approx(expected)
    assert score(query, document, 'bm25', corpus=corpus) == pytest.approx(expected)
    assert simdif(query, document, ['bm25'], corpus=corpus) == {'bm25': pytest.approx(expected)}
    assert simdif(query, document, ['okapi'], corpus=corpus) == {'okapi': pytest.approx(expected)}
