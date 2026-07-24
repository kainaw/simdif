import pytest
from simdif.metrics.soundex import sim_soundex, dif_soundex, _get_soundex
from simdif import soundex, sim, simdif

def test_soundex():
    # Edge case: both sides empty => phonetically "identical" => 1.0.
    assert sim_soundex("", "") == 1.0
    # One side empty => 0.0.
    assert sim_soundex("", "Robert") == 0.0
    # Identical words share a code.
    assert sim_soundex("Robert", "Robert") == 1.0
    # Classic Soundex example: "Robert" and "Rupert" both code to R163.
    #   Robert -> R + b(1) r(6) t(3) = R163
    #   Rupert -> R + p(1) r(6) t(3) = R163
    assert sim_soundex("Robert", "Rupert") == 1.0
    # Different first letter and code => 0.0.
    #   Smith -> S + m(5) t(3) = S530
    assert sim_soundex("Robert", "Smith") == 0.0
    # Vowel-reset rule (standard American Soundex): a repeated code separated
    # by a vowel is recorded twice, not collapsed. "Ababa" -> A110 (both b's
    # are coded), whereas "Aba" -> A100. They must therefore differ.
    assert _get_soundex("Ababa") == ("A", "110")
    assert _get_soundex("Aba") == ("A", "100")
    assert sim_soundex("Ababa", "Aba") == 0.0
    # H and W stay transparent (they neither code nor reset): "Smith" -> S530.
    assert _get_soundex("Smith") == ("S", "530")
    # dif is the complement of sim.
    assert dif_soundex("Robert", "Rupert") == 0.0
    assert dif_soundex("Robert", "Smith") == 1.0
    # Convenience name (default 'sim' role) and dispatchers.
    assert soundex("Robert", "Rupert") == 1.0
    assert sim("Robert", "Rupert", "soundex") == 1.0
    assert simdif("Robert", "Rupert", ["soundex"]) == {"soundex": 1.0}
