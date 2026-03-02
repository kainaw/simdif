"""
simdif - Similarity, Difference, Distance, and Score Metrics for Python
=======================================================================

A unified library for computing similarity, difference, distance, alignment
scores, and sequence traces across sets, vectors, strings, and sequences.

Supported metric categories:
    sim_*   - Similarity metrics (e.g., Jaccard, Cosine, Dice)
    dif_*   - Difference metrics (e.g., Levenshtein, Hamming)
    dist_*  - Distance metrics (e.g., Euclidean, Manhattan)
    score_* - Alignment scores (e.g., Needleman-Wunsch, Smith-Waterman)
    trace_* - Alignment tracebacks (aligned sequence pairs)

Basic usage:
    from simdif import sim, dist
    sim("night", "nacht", "jaro")
    dist([1, 2, 3], [4, 5, 6], "euclidean")

Author: C. Shaun Wagner
License: MIT
"""

import math
import numbers
import sys
import os
_DEFINITIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "definitions.txt")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

VALID_PREFIXES = {'sim', 'dif', 'dist', 'score', 'trace','matrix', 'explain', 'info'}


def _resolve_metric(name: str):
    name = name.lower().replace('-','_')
    if '_' in name:
        prefix, base = name.split('_', 1)
        if prefix in VALID_PREFIXES:
            entry = METRICS.get(base)
            if not entry or prefix not in entry:
                raise ValueError(f"Unknown metric '{name}'")
            return prefix, entry[prefix], base
    entry = METRICS.get(name)
    if not entry:
        raise ValueError(f"Unknown metric '{name}'")
    role = entry['default']
    return role, entry[role], name


def simdif(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: simdif(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric(metric)
    return func(a, b, **kwargs)


def sim(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: sim(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("sim_"+metric)
    if role != 'sim':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a similarity metric")
    return func(a, b, **kwargs)


def dif(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: dif(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("dif_"+metric)
    if role != 'dif':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a difference metric")
    return func(a, b, **kwargs)


def dist(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: dist(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("dist_"+metric)
    if role != 'dist':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a distance metric")
    return func(a, b, **kwargs)


def score(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: score(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("score_"+metric)
    if role != 'score':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a scoring metric")
    return func(a, b, **kwargs)


def trace(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: trace(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("trace_"+metric)
    if role != 'trace':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a trace metric")
    return func(a, b, **kwargs)


def matrix(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: matrix(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("matrix_"+metric)
    if role != 'matrix':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a trace metric")
    return func(a, b, **kwargs)


def info(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: info(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("info_"+metric)
    if role != 'info':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a trace metric")
    return func(a, b, **kwargs)


def explain(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: explain(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric("explain_"+metric)
    if role != 'explain':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a trace metric")
    return func(a, b, **kwargs)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def to_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, (str, bytes)):
        return list(val)
    if isinstance(val, dict):
        return list(val.values())
    if 'numpy' in sys.modules:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.flatten().tolist()
    if 'pandas' in sys.modules:
        import pandas as pd
        if isinstance(val, pd.Series):
            return val.tolist()
    if 'torch' in sys.modules:
        import torch
        if isinstance(val, torch.Tensor):
            return val.flatten().tolist()
    try:
        return list(val)
    except TypeError:
        return [val]

def _make_hashable(x):
    if isinstance(x, (list, tuple)):
        return tuple(_make_hashable(i) for i in x)
    if isinstance(x, (set, frozenset)):
        return frozenset(x)
    if isinstance(x, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
    return x
    
def to_set(val):
    if val is None:
        return set()
    if isinstance(val, set):
        return val
    lst = to_list(val)
    try:
        return set(lst)
    except TypeError:
        return {_make_hashable(x) for x in lst}

def to_list_numeric(val, allow_complex=False) -> list:
    lst = to_list(val)
    check = numbers.Number if allow_complex else numbers.Real
    if not all(isinstance(x, check) for x in lst):
        raise TypeError(f"Expected numeric values, got {[type(x).__name__ for x in lst if not isinstance(x, check)]}")
    return lst

def to_binary(val, width=None) -> list:
    if not isinstance(val, int):
        raise TypeError(f"to_binary expects an int, got {type(val).__name__}")
    bits = bin(val)[2:]  # strip '0b' prefix
    if width is not None:
        bits = bits.zfill(width)
    return [int(b) for b in bits]

def to_tokens(val):
    if val is None:
        return []
    if isinstance(val, (str, bytes)):
        return val.split()
    return to_list(val)

def to_distribution(val):
    lst = to_list_numeric(val)
    if any(x < 0 for x in lst):
        raise ValueError("Distribution contains negative values")
    total = sum(lst)
    if total == 0:
        raise ValueError("Distribution sums to zero")
    return [x / total for x in lst]

def _rank(lst):
    sorted_with_index = sorted(enumerate(lst), key=lambda x: x[1])
    ranks = [0.0] * len(lst)
    i = 0
    while i < len(lst):
        j = i
        while j < len(lst) - 1 and sorted_with_index[j+1][1] == sorted_with_index[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j+1):
            ranks[sorted_with_index[k][0]] = avg_rank
        i = j + 1
    return ranks


def _aleph_counts(a, b, n_universe=0):
    a, b = to_set(a), to_set(b)
    n11 = len(a & b)
    n10 = len(a - b)
    n01 = len(b - a)
    n00 = max(0, n_universe - len(a | b))
    return n00, n01, n10, n11


# ------------------------------------------------------------------
# Vector Metrics
# ------------------------------------------------------------------

def dist_hamming(a, b, binary=False) -> int:
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list(a), to_list(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x != y for x, y in zip(a, b))

def sim_hamming(a, b, binary=False) -> float:
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list(a), to_list(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0
    return 1 - (dist_hamming(a, b) / len(a))

def dif_hamming(a, b, binary=False) -> float:
    return 1 - sim_hamming(a, b, binary)

def sim_tanimoto(a, b, binary=False) -> float:
    if binary:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("binary=True requires integer inputs")
        width = max(a.bit_length(), b.bit_length())
        a, b = to_binary(a, width), to_binary(b, width)
    else:
        a, b = to_list(a), to_list(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0
    intersection = sum(x == y == 1 for x, y in zip(a, b))
    return intersection / (sum(a) + sum(b) - intersection)

def dif_tanimoto(a, b, binary=False) -> float:
    return 1 - sim_tanimoto(a, b, binary)

# ------------------------------------------------------------------
# Distance metrics
# ------------------------------------------------------------------

def dist_minkowski(a, b, p=None) -> float:
    if p is None:
        raise TypeError("dist_minkowski() missing 1 required positional argument: 'p'")
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if 'scipy' in sys.modules:
        from scipy.spatial import distance
        return float(distance.minkowski(a, b, p))
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1/p)

dist_euclidean = lambda a, b: dist_minkowski(a, b, p=2)
dist_manhattan = lambda a, b: dist_minkowski(a, b, p=1)
dist_taxicab = dist_manhattan
dist_cityblock = dist_manhattan

def dist_chebyshev(a, b) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b): raise ValueError("Length mismatch")
    return max(abs(x - y) for x, y in zip(a, b))
    
dist_chessboard = dist_chebyshev
dist_linf = dist_chebyshev

def dist_canberra(a, b) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b): 
        raise ValueError("Length mismatch")
    score = 0.0
    for x, y in zip(a, b):
        denominator = abs(x) + abs(y)
        if denominator > 0:
            score += abs(x - y) / denominator
    return score


def dist_kl_divergence(a, b) -> float:
    a = _to_distribution(a, "a")
    b = _to_distribution(b, "b")
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    return sum(p * math.log(p / q) for p, q in zip(a, b) if p > 0)
dist_kullback_leibler=dist_kl_divergence

def dist_js_divergence(a, b) -> float:
    a = _to_distribution(a, "a")
    b = _to_distribution(b, "b")
    if len(a) != len(b):
        raise ValueError(f"Distributions must have the same length, got {len(a)} and {len(b)}")
    m = [(x + y) / 2 for x, y in zip(a, b)]
    kl_a = sum(p * math.log(p / q) for p, q in zip(a, m) if p > 0)
    kl_b = sum(p * math.log(p / q) for p, q in zip(b, m) if p > 0)
    return (kl_a + kl_b) / 2
dist_jensen_shannon = dist_js_divergence

# ------------------------------------------------------------------
# Edit Distance Metrics
# ------------------------------------------------------------------

def _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=None, match_score=None, local=False, maximize=False) -> list:
    rows = len(s1) + 1
    cols = len(s2) + 1
    matrix = [[0] * cols for _ in range(rows)]
    if not local:
        for i in range(rows): matrix[i][0] = i * delete
        for j in range(cols): matrix[0][j] = j * insert
    for i in range(1, rows):
        for j in range(1, cols):
            options = [
                matrix[i-1][j] + delete,
                matrix[i][j-1] + insert,
            ]
            if substitute is not None:
                diag = matrix[i-1][j-1]
                if s1[i-1] == s2[j-1]:
                    options.append(diag if match_score is None else diag + match_score)
                else:
                    options.append(diag + substitute)
            if transpose is not None and i > 1 and j > 1:
                if s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    options.append(matrix[i-2][j-2] + transpose)
            cell = max(options) if maximize else min(options)
            matrix[i][j] = max(0, cell) if local else cell
    return matrix

def dist_levenshtein(a, b) -> int:
    if isinstance(a, str) and isinstance(b, str) and 'Levenshtein' in sys.modules:
        return float(sys.modules['Levenshtein'].distance(a, b))
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=None, local=False, maximize=False)[-1][-1]

def dist_indel(a, b) -> int:
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=None, transpose=None, local=False, maximize=False)[-1][-1]

def dist_osa(a, b) -> int:
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=1, local=False, maximize=False)[-1][-1]

def score_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    return _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)[-1][-1]

score_needleman = score_needleman_wunsch
score_wunsch = score_needleman_wunsch

def score_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=True, maximize=True)
    return max(matrix[i][j] for i in range(len(s1)+1) for j in range(len(s2)+1))

score_smith = score_smith_waterman
score_waterman = score_smith_waterman

def sim_levenshtein(a, b) -> float:
    s1, s2 = to_list(a), to_list(b)
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 1.0
    return 1 - (dist_levenshtein(s1, s2) / max_len)

def dif_levenshtein(a, b) -> float:
    return 1 - sim_levenshtein(a, b)

def sim_monge_elkan(a, b, method="jaro_winkler") -> float:
    tokens_a = to_tokens(a)
    tokens_b = to_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    total_score = 0.0
    for s in tokens_a:
        best_match = max(sim(s, t, method) for t in tokens_b)
        total_score += best_match
    return total_score / len(tokens_a)

def _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, gap_symbol="-"):
    rows, cols = len(s1), len(s2)
    if local:
        curr_i, curr_j = 0, 0
        max_val = -float('inf')
        for r in range(rows + 1):
            for c in range(cols + 1):
                if matrix[r][c] >= max_val:
                    max_val = matrix[r][c]
                    curr_i, curr_j = r, c
    else:
        curr_i, curr_j = rows, cols
    align1, align2 = [], []
    while curr_i > 0 or curr_j > 0:
        if local and matrix[curr_i][curr_j] == 0:
            break
        current_val = matrix[curr_i][curr_j]
        if curr_i > 0 and curr_j > 0:
            score = match_score if s1[curr_i-1] == s2[curr_j-1] else mismatch_penalty
            if current_val == matrix[curr_i-1][curr_j-1] + score:
                align1.append(s1[curr_i-1])
                align2.append(s2[curr_j-1])
                curr_i -= 1
                curr_j -= 1
                continue
        if curr_i > 0 and current_val == matrix[curr_i-1][curr_j] + gap_penalty:
            align1.append(s1[curr_i-1])
            align2.append(gap_symbol)
            curr_i -= 1
        else:
            align1.append(gap_symbol)
            align2.append(s2[curr_j-1])
            curr_j -= 1
    return align1[::-1], align2[::-1]

def trace_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, gap_symbol=gap_symbol)

def trace_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=True, maximize=True)
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=True, gap_symbol=gap_symbol)

def dist_lee(a, b, q=None):
    a, b = to_list_numeric(a), to_list_numeric(b)
    if q is None:
        q = max(max(a), max(b)) + 1
    distance = 0
    for va, vb in zip(a, b):
        diff = abs(va - vb)
        distance += min(diff, q - diff)
    return distance

def _fill_dp_matrix(a, b, **kwargs):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, **kwargs)
    header_row = [" ", " "] + [str(x) for x in s2]
    side_labels = [" ", " "] + [str(x) for x in s1]
    for i, row in enumerate(matrix):
        row.insert(0, side_labels[i+1])
    matrix.insert(0, header_row)
    return matrix

def matrix_levenshtein(a, b):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=1, transpose=None, local=False, maximize=False)

def matrix_osa(a, b):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=1, transpose=1, local=False, maximize=False)

def matrix_indel(a, b):
    return _fill_dp_matrix(a, b, insert=1, delete=1, substitute=None, transpose=None, local=False, maximize=False)

def matrix_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    return _fill_dp_matrix(a, b, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=True, maximize=True)

def matrix_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
    return _fill_dp_matrix(a, b, insert=gap_penalty, delete=gap_penalty, substitute=mismatch_penalty, match_score=match_score, local=False, maximize=True)

def matrix_lcs(a, b):
    return _fill_dp_matrix(a, b, insert=0, delete=0, substitute=None, match_score=1, local=False, maximize=True)
# ------------------------------------------------------------------
# Sequence metrics
# ------------------------------------------------------------------

def score_lcs(a, b) -> int:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, 1, 0, 0, local=False, maximize=True)
    return matrix[-1][-1]

def dist_lcs(a, b) -> int:
    s1, s2 = to_list(a), to_list(b)
    return len(s1) + len(s2) - 2 * score_lcs(s1, s2)

def sim_jaro(a, b) -> float:
    s1, s2 = to_list(a), to_list(b)
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0: return 1.0
    if len1 == 0 or len2 == 0: return 0.0

    match_window = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0: return 0.0

    # Count transpositions
    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]: k += 1
            if s1[i] != s2[k]: transpositions += 1
            k += 1
            
    return (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3

def dif_jaro(a, b) -> float:
    return 1.0 - sim_jaro(a, b)

def sim_jaro_winkler(a, b, p=0.1, max_l=4) -> float:
    if p > 0.25:
        raise ValueError("p should not exceed 0.25 to keep score within [0, 1]")
    s1, s2 = to_list(a), to_list(b)
    j = sim_jaro(s1, s2)
    l = 0
    for char1, char2 in zip(s1[:max_l], s2[:max_l]):
        if char1 == char2: l += 1
        else: break
    return j + (l * p * (1 - j))

def dif_jaro_winkler(a, b, p=0.1, max_l=4) -> float:
    return 1 - sim_jaro_winkler(a, b, p, max_l)

# ------------------------------------------------------------------
# Frequency/Abundance Metrics (for Dicts or Counters)
# ------------------------------------------------------------------

def dist_bray_curtis(a, b) -> float:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError("Vector length mismatch")
    
    diff_sum = sum(abs(x - y) for x, y in zip(v1, v2))
    total_sum = sum(abs(x + y) for x, y in zip(v1, v2))
    
    return diff_sum / total_sum if total_sum != 0 else 0.0


METRICS = {
    'hamming': {
        'default': 'dist',
        'dist': dist_hamming,
        'sim': sim_hamming,
        'dif': dif_hamming,
    },
    'tanimoto': {
        'default': 'sim',
        'sim': sim_tanimoto,
        'dif': dif_tanimoto,
    },

    'minkowski': {
        'default': 'dist',
        'dist': dist_minkowski,
    },
    'euclidean': {
        'default': 'dist',
        'dist': dist_euclidean,
    },
    'manhattan': {
        'default': 'dist',
        'dist': dist_manhattan,
    },
    'chebyshev': {
        'default': 'dist',
        'dist': dist_chebyshev,
    },
    'canberra': {
        'default':'dist',
        'dist': dist_canberra,
    },

    'kl_divergence': {
        'default':'dist',
        'dist': dist_kl_divergence,
    },
    'kullback_leibler': {
        'default':'dist',
        'dist': dist_kl_divergence,
    },

    'js_divergence': {
        'default':'dist',
        'dist': dist_js_divergence,
    },
    'jensen_shannon': {
        'default':'dist',
        'dist': dist_js_divergence,
    },

    'levenshtein': {
        'default': 'dist',
        'dist': dist_levenshtein,
        'sim': sim_levenshtein,
        'dif': dif_levenshtein,
        'matrix': matrix_levenshtein,
    },

    'needleman_wunsch': {
        'default': 'score',
        'score': score_needleman_wunsch,
        'trace': trace_needleman_wunsch,
        'matrix': matrix_needleman_wunsch,
    },

    'smith_waterman': {
        'default': 'score',
        'score': score_smith_waterman,
        'trace': trace_smith_waterman,
        'matrix': matrix_smith_waterman,
    },
    
    'monge_elkan': {
        'default': 'sim',
        'sim': sim_monge_elkan,
    },

    'osa': {
        'default': 'dist',
        'dist': dist_osa,
     },

    'lee': {
        'default': 'dist',
        'dist': dist_lee,
    },

    'lcs': {
        'default': 'score',
        'score': score_lcs,
        'dist': dist_lcs,
        'matrix': matrix_lcs,
    },

    'jaro': {
        'default': 'sim',
        'sim': sim_jaro,
        'dif': dif_jaro,
    },

    'jaro_winkler': {
        'default': 'sim',
        'sim': sim_jaro_winkler,
        'dif': dif_jaro_winkler,
    },

    'bray_curtis': {
        'default': 'dist',
        'dist': dist_bray_curtis,
    },
}


# Similarity is the shadow two things cast in the same light. 🦔
