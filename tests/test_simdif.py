import pytest
import simdif


# ---------------------------------------------------------
# Set Metrics
# ---------------------------------------------------------

def test_sim_jaccard_basic():
    assert simdif.sim_jaccard({1, 2}, {2, 3}) == pytest.approx(1/3)
    assert simdif.sim_jaccard([], []) == 1.0
    assert simdif.sim_jaccard([], [1]) == 0.0


def test_sim_dice_basic():
    assert simdif.sim_dice({1, 2}, {2, 3}) == pytest.approx(2/4)
    assert simdif.sim_dice([], []) == 1.0
    assert simdif.sim_dice([], [1]) == 0.0


def test_sim_overlap_basic():
    assert simdif.sim_overlap({1, 2}, {2, 3}) == pytest.approx(1/2)
    assert simdif.sim_overlap([], []) == 1.0
    assert simdif.sim_overlap([], [1]) == 0.0


def test_sim_tversky_basic():
    assert simdif.sim_tversky({1, 2}, {2, 3}, alpha=0.5, beta=0.5) == pytest.approx(0.5)


# ---------------------------------------------------------
# Vector Metrics
# ---------------------------------------------------------

def test_sim_cosine_basic():
    assert simdif.sim_cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert simdif.sim_cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert simdif.sim_cosine([1, 1], [1, 1]) == pytest.approx(1.0)


def test_sim_cosine_length_mismatch():
    with pytest.raises(ValueError):
        simdif.sim_cosine([1, 2], [1])


def test_dist_hamming_basic():
    assert simdif.dist_hamming([1, 0, 1], [1, 1, 1]) == 1
    assert simdif.sim_hamming([1, 0, 1], [1, 1, 1]) == pytest.approx(2/3)


def test_dist_hamming_binary():
    assert simdif.dist_hamming(0b1010, 0b1110, binary=True) == 1


# ---------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------

def test_dist_euclidean_basic():
    assert simdif.dist_euclidean([0, 0], [3, 4]) == 5.0


def test_dist_manhattan_basic():
    assert simdif.dist_manhattan([1, 2], [4, 6]) == 7


def test_dist_chebyshev_basic():
    assert simdif.dist_chebyshev([1, 5], [4, 1]) == 4


# ---------------------------------------------------------
# Edit Distance Metrics
# ---------------------------------------------------------

def test_dist_levenshtein_basic():
    assert simdif.dist_levenshtein("kitten", "sitting") == 3
    assert simdif.sim_levenshtein("abc", "abc") == 1.0
    assert simdif.sim_levenshtein("abc", "axc") == pytest.approx(2/3)


def test_score_needleman_wunsch_basic():
    score = simdif.score_needleman_wunsch("GATTACA", "GCATGCU")
    assert isinstance(score, int)


def test_score_smith_waterman_basic():
    score = simdif.score_smith_waterman("GATTACA", "GCATGCU")
    assert isinstance(score, int)


def test_trace_needleman_wunsch_basic():
    a, b = simdif.trace_needleman_wunsch("GATTACA", "GCATGCU")
    assert len(a) == len(b)


def test_trace_smith_waterman_basic():
    a, b = simdif.trace_smith_waterman("GATTACA", "GCATGCU")
    assert len(a) == len(b)


# ---------------------------------------------------------
# Sequence Metrics
# ---------------------------------------------------------

def test_score_lcs_basic():
    assert simdif.score_lcs("abcde", "ace") == 3
    assert simdif.dist_lcs("abcde", "ace") == 5 + 3 - 2*3  # formula check


def test_sim_jaro_basic():
    assert simdif.sim_jaro("MARTHA", "MARHTA") > 0.9
    assert simdif.sim_jaro("DWAYNE", "DUANE") > 0.7


def test_sim_jarowinkler_basic():
    assert simdif.sim_jarowinkler("MARTHA", "MARHTA") > 0.9


# ---------------------------------------------------------
# Frequency Metrics
# ---------------------------------------------------------

def test_dist_braycurtis_basic():
    assert simdif.dist_braycurtis([1, 2], [1, 2]) == 0.0
    assert simdif.dist_braycurtis([1, 2], [2, 4]) == pytest.approx(1/3)


# ---------------------------------------------------------
# High-level dispatch functions
# ---------------------------------------------------------

def test_simdif_dispatch_single():
    assert simdif.simdif([1, 2], [2, 3], "jaccard") == pytest.approx(1/3)


def test_simdif_dispatch_multiple():
    result = simdif.simdif([1, 2], [2, 3], ["jaccard", "dice"])
    assert "jaccard" in result
    assert "dice" in result


def test_sim_dispatch_error():
    with pytest.raises(ValueError):
        simdif.sim([1], [1], "levenshtein")  # levenshtein default is dist


def test_dist_dispatch_error():
    with pytest.raises(ValueError):
        simdif.dist([1], [1], "jaccard")  # jaccard default is sim
