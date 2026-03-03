import pytest
from simdif import to_set, to_list, to_list_numeric, to_list_numeric_aligned

def test_simdif():
    assert to_set("abcb") == {'a', 'b', 'c'}
    assert to_list("abcb") == ['a', 'b', 'c', 'b']
    assert to_list_numeric(["1", "2", "3"]) == [1, 2, 3]
    assert to_list_numeric_aligned([1, 2, 3],[4, 5], pad_value="6") == ([1, 2, 3], [4, 5, 6])
