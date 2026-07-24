import math
import pytest
from simdif.metrics.js_divergence import dist_js_divergence, sim_js_divergence
from simdif import js_divergence, dist, sim, simdif

def test_js_divergence():
    # Identical distributions -> divergence 0, similarity 1.
    assert dist_js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert sim_js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    # Disjoint support: M = [0.5, 0.5]; KL_P = KL_Q = ln(2); JSD = ln(2) (the max).
    assert dist_js_divergence([1, 0], [0, 1]) == pytest.approx(math.log(2))
    assert sim_js_divergence([1, 0], [0, 1]) == pytest.approx(1.0 / (1.0 + math.log(2)))
    # Known value: P=[0.5,0.5], Q=[1,0], M=[0.75,0.25].
    kl_p = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    kl_q = 1 * math.log(1 / 0.75)  # Q's zero bin contributes 0 (skipped when p == 0).
    expected = (kl_p + kl_q) / 2
    assert dist_js_divergence([0.5, 0.5], [1, 0]) == pytest.approx(expected)
    # Symmetry.
    assert dist_js_divergence([0.5, 0.5], [1, 0]) == pytest.approx(dist_js_divergence([1, 0], [0.5, 0.5]))
    # Convenience name (default role is 'dist') + role dispatchers + simdif dict form.
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(math.log(2))
    assert dist([1, 0], [0, 1], 'js_divergence') == pytest.approx(math.log(2))
    assert sim([1, 0], [0, 1], 'js_divergence') == pytest.approx(1.0 / (1.0 + math.log(2)))
    assert simdif([1, 0], [0, 1], ['js_divergence']) == {'js_divergence': pytest.approx(math.log(2))}
    # Alias registered under 'jensen_shannon'.
    assert simdif([1, 0], [0, 1], ['jensen_shannon']) == {'jensen_shannon': pytest.approx(math.log(2))}
