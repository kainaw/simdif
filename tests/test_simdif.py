import pytest
from simdif import to_set, to_list, to_list_numeric, to_list_numeric_aligned
from simdif import info, explain, simdif, sim
from simdif.simdif import METRICS

def test_simdif():
    assert to_set("abcb") == {'a', 'b', 'c'}
    assert to_list("abcb") == ['a', 'b', 'c', 'b']
    assert to_list_numeric(["1", "2", "3"]) == [1, 2, 3]
    assert to_list_numeric_aligned([1, 2, 3],[4, 5], pad_value="6") == ([1, 2, 3], [4, 5, 6])


def test_info_dispatcher():
    # info_* take no arguments -- they are pure documentation. The dispatcher
    # used to forward (a, b, **kwargs) to them, so every call raised TypeError.
    assert info('chebyshev').startswith('Chebyshev Distance')
    assert info('linf') == info('chebyshev')            # aliases resolve
    # The (a, b, metric) form is accepted for uniformity with sim()/dist();
    # a, b and kwargs are ignored rather than forwarded.
    assert info(None, None, 'chebyshev') == info('chebyshev')
    assert info([1, 2], [3, 4], 'jaccard', pad_value=0) == info('jaccard')
    assert info(['chebyshev', 'jaccard']) == {'chebyshev': info('chebyshev'),
                                              'jaccard': info('jaccard')}
    assert simdif([1, 2], [3, 4], 'info_chebyshev') == info('chebyshev')
    with pytest.raises(ValueError, match="requires a metric name"):
        info()
    with pytest.raises(ValueError, match="Unknown metric"):
        info('not_a_metric')


def test_every_registered_metric_documents_itself():
    """info() must return text for every name in METRICS, aliases included.

    This caught info_index_of_dissimilarity calling .trim() instead of .strip().
    """
    for name in METRICS:
        text = info(name)
        assert isinstance(text, str) and text.strip(), name


def test_every_registered_metric_explains_itself():
    """explain() must not crash on valid input for its class.

    This caught explain_cosine referencing an undefined dif_cosine,
    explain_tversky doing set subtraction on already-stringified lists, and
    explain_smc requiring an n_universe that sim_smc defaults to None.
    """
    numeric = ([1, 2, 3], [2, 3, 4])
    # The geodesic family is registered as 'vector' but validates its input as
    # exactly one [latitude, longitude] pair, so it needs its own sample.
    lat_lon = ([40.7, -74.0], [51.5, -0.1])
    special = {'geodesic', 'earth'}
    for name in METRICS:
        a, b = lat_lon if name in special else numeric
        out = explain(a, b, name)
        assert isinstance(out, str) and out.strip(), name


def test_explain_agrees_with_the_metric_it_explains():
    # explain_tversky divided by the union instead of |A∩B| + a|A-B| + b|B-A|,
    # so its printed similarity disagreed with sim_tversky.
    a, b = ['a', 'b', 'c'], ['b', 'c', 'd']
    assert sim(a, b, 'tversky') == pytest.approx(2 / 3)
    assert "Similarity: 0.6667" in explain(a, b, 'tversky')
    assert "2 / (2 + 0.5×1 + 0.5×1)" in explain(a, b, 'tversky')
