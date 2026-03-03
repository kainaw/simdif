import pytest
from simdif.metrics.kendall_tau import sim_kendall_tau, dist_kendall_tau
from simdif import kendall_tau, sim, simdif

def test_kendall_tau():
    assert sim_kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert sim_kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert sim_kendall_tau([1, 2, 3], [2, 1, 3]) == pytest.approx(1/3)
    assert sim_kendall_tau([1,2,2], [1,2,2]) == pytest.approx(0.6666666666666666)
    with pytest.raises(ValueError, match="at least 2 elements"):
        sim_kendall_tau([1], [1])
    a = [1, 5, 2, 10]
    b = [2, 4, 1, 9]
    s = sim_kendall_tau(a, b)
    d = dist_kendall_tau(a, b)
    assert s + d == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert sim([1, 2, 3, 4], [1, 2, 3, 4], 'kendall_tau') == pytest.approx(1.0)
    assert simdif([1, 2, 3, 4], [1, 2, 3, 4], ['kendall_tau','tau_a']) == {
        'kendall_tau': pytest.approx(1.0),
        'tau_a': pytest.approx(1.0)
    }
