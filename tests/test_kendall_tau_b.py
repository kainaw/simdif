import pytest
from simdif.metrics.kendall_tau_b import sim_kendall_tau_b, dist_kendall_tau_b
from simdif import kendall_tau_b, sim, simdif

def test_kendall_tau_b():
    assert sim_kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert sim_kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert sim_kendall_tau_b([1, 2, 3], [2, 1, 3]) == pytest.approx(1/3)
    assert sim_kendall_tau_b([1,2,2], [1,2,2]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 2 elements"):
        sim_kendall_tau_b([1], [1])
    a = [1, 5, 2, 10]
    b = [2, 4, 1, 9]
    s = sim_kendall_tau_b(a, b)
    d = dist_kendall_tau_b(a, b)
    assert s + d == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert sim([1, 2, 3, 4], [1, 2, 3, 4], 'kendall_tau_b') == pytest.approx(1.0)
    assert simdif([1, 2, 3, 4], [1, 2, 3, 4], ['kendall_tau_b','tau_b']) == {
        'kendall_tau_b': pytest.approx(1.0),
        'tau_b': pytest.approx(1.0)
    }
