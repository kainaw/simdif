import math
import pytest
from simdif.metrics.kl_divergence import dist_kl_divergence, sim_kl_divergence
from simdif import kl_divergence, dist, sim, simdif

def test_kl_divergence():
    # Identical distributions -> divergence 0, similarity 1.
    assert dist_kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert sim_kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)
    # Known value: sum p*log(p/q) with no zero in q where p>0.
    # 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75).
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert dist_kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)
    assert sim_kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(1.0 / (1.0 + expected))
    # Asymmetry: D_KL(P||Q) != D_KL(Q||P).
    assert dist_kl_divergence([0.5, 0.5], [0.25, 0.75]) != pytest.approx(
        dist_kl_divergence([0.25, 0.75], [0.5, 0.5]))
    # When Q has a zero where P>0 the divergence is infinite (standard KL
    # convention); similarity 1/(1+inf) collapses to 0.0.
    assert dist_kl_divergence([1, 0], [0, 1]) == math.inf
    assert sim_kl_divergence([1, 0], [0, 1]) == pytest.approx(0.0)
    # Convenience name (default role is 'dist') + role dispatchers + simdif dict form.
    assert kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)
    assert dist([0.5, 0.5], [0.25, 0.75], 'kl_divergence') == pytest.approx(expected)
    assert sim([0.5, 0.5], [0.25, 0.75], 'kl_divergence') == pytest.approx(1.0 / (1.0 + expected))
    assert simdif([0.5, 0.5], [0.25, 0.75], ['kl_divergence']) == {'kl_divergence': pytest.approx(expected)}
    # Alias registered under 'kullback_leibler'.
    assert simdif([0.5, 0.5], [0.25, 0.75], ['kullback_leibler']) == {'kullback_leibler': pytest.approx(expected)}
