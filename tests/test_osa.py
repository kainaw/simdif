import pytest
from simdif.metrics.osa import dist_osa
from simdif import osa, dist, simdif

def test_osa():
    assert dist_osa("CA", "ABC") == 3.0
    assert dist_osa("CA", "ABCA") == 2.0
    assert dist_osa("BCA", "ABC") == 2.0
    assert dist_osa("ABC", "CDE") <= dist_osa("ABC", "BCD") + dist_osa("BCD", "CDE")
    assert osa("CA", "ABC") == 3.0
    assert dist("CA", "ABC","osa") == 3.0
    assert simdif("CA","ABC",["osa"]) == {"osa": 3.0}
