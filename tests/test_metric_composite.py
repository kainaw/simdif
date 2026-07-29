import pytest
from simdif import sim_jaccard, dist_hamming, Metric


A, B = ['a', 'b', 'c'], ['b', 'c', 'd']


def test_composite_is_a_metric():
    mysim = 2 * sim_jaccard - 0.5 * dist_hamming
    assert isinstance(mysim, Metric)
    expected = 2 * sim_jaccard(A, B) - 0.5 * dist_hamming(A, B)
    assert mysim(A, B) == pytest.approx(expected)


def test_add_sub_mul_truediv_between_metrics():
    assert (sim_jaccard + dist_hamming)(A, B) == pytest.approx(sim_jaccard(A, B) + dist_hamming(A, B))
    assert (sim_jaccard - dist_hamming)(A, B) == pytest.approx(sim_jaccard(A, B) - dist_hamming(A, B))
    assert (sim_jaccard * dist_hamming)(A, B) == pytest.approx(sim_jaccard(A, B) * dist_hamming(A, B))
    assert (sim_jaccard / dist_hamming)(A, B) == pytest.approx(sim_jaccard(A, B) / dist_hamming(A, B))


def test_metric_with_plain_number_operand():
    assert (sim_jaccard + 1)(A, B) == pytest.approx(sim_jaccard(A, B) + 1)
    assert (1 + sim_jaccard)(A, B) == pytest.approx(1 + sim_jaccard(A, B))
    assert (sim_jaccard - 1)(A, B) == pytest.approx(sim_jaccard(A, B) - 1)
    assert (1 - sim_jaccard)(A, B) == pytest.approx(1 - sim_jaccard(A, B))
    assert (sim_jaccard * 2)(A, B) == pytest.approx(sim_jaccard(A, B) * 2)
    assert (2 * sim_jaccard)(A, B) == pytest.approx(2 * sim_jaccard(A, B))
    assert (sim_jaccard / 2)(A, B) == pytest.approx(sim_jaccard(A, B) / 2)
    assert (4 / sim_jaccard)(A, B) == pytest.approx(4 / sim_jaccard(A, B))


def test_unary_negation():
    assert (-sim_jaccard)(A, B) == pytest.approx(-sim_jaccard(A, B))


def test_metric_with_plain_callable_operand():
    # Any callable(a, b, **kwargs) -> number can be mixed in, not just @Metric functions.
    def always_half(a, b, **kwargs):
        return 0.5
    mysim = sim_jaccard - always_half
    assert mysim(A, B) == pytest.approx(sim_jaccard(A, B) - 0.5)


def test_composite_forwards_kwargs():
    # dist_hamming needs padding to compare unequal-length sequences; the
    # composite must forward **kwargs to every operand, not just call().
    mysim = sim_jaccard - dist_hamming
    a, b = ['a', 'b'], ['a', 'b', 'c']
    expected = sim_jaccard(a, b) - dist_hamming(a, b, pad_value='')
    assert mysim(a, b, pad_value='') == pytest.approx(expected)


def test_chained_composition_stays_a_metric():
    mysim = (2 * sim_jaccard - 0.5 * dist_hamming) + 1
    assert isinstance(mysim, Metric)
    expected = 2 * sim_jaccard(A, B) - 0.5 * dist_hamming(A, B) + 1
    assert mysim(A, B) == pytest.approx(expected)
