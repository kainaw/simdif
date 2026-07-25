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
from functools import wraps


# ------------------------------------------------------------------
# Composite Metric Functionality
# ------------------------------------------------------------------

class Metric:
    """
    This allows you to build composite metrics like:
    myfunc = 2*sim_jaccard - sim_cosine + 0.4 dist_hamming
    """
    def __init__(self, func):
        self.func = func
        wraps(func)(self)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def _wrap_operand(self, other):
        if isinstance(other, Metric) or callable(other):
            return other
        return lambda *args, **kwargs: other
    def __add__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: self(*a, **k) + other_fn(*a, **k))
    def __sub__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: self(*a, **k) - other_fn(*a, **k))
    def __mul__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: self(*a, **k) * other_fn(*a, **k))
    def __truediv__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: self(*a, **k) / other_fn(*a, **k))
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: other_fn(*a, **k) - self(*a, **k))
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other):
        other_fn = self._wrap_operand(other)
        return Metric(lambda *a, **k: other_fn(*a, **k) / self(*a, **k))


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

def to_list(val, **kwargs):
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


def to_list_numeric(val, **kwargs) -> list:
    allow_complex = kwargs.get("allow_complex", False)
    ascii_mode = kwargs.get("ascii", False)
    raw_list = to_list(val)
    numeric_vector = []
    target_type = numbers.Number if allow_complex else numbers.Real
    for item in raw_list:
        try:
            num = float(item)
            numeric_vector.append(num)
            continue
        except (ValueError, TypeError):
            if allow_complex:
                try:
                    num = complex(item)
                    numeric_vector.append(num)
                    continue
                except (ValueError, TypeError):
                    pass
        text_rep = str(item)
        if ascii_mode:
            numeric_vector.extend([float(ord(c)) for c in text_rep])
        else:
            numeric_vector.append(0.0)
    return numeric_vector


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
    min_val = min(lst)
    if min_val < 0: # Make all values positive
        ep = 1e-12 # Tiny shift to avoid zero
        lst = [(x - min_val) + ep for x in lst]
    total = sum(lst)
    if total == 0:
        raise ValueError("Distribution sums to zero")
    return [x / total for x in lst]

def to_qgram(val, n=2, pad=None):
    """
    Break a sequence into contiguous n-grams ("shingles"). This is the
    generic tokenizer behind q-gram/n-gram based metrics -- compose it
    with existing set/vector metrics rather than reaching for a
    dedicated q-gram function:
        sim_jaccard(to_qgram(a), to_qgram(b))       # Jaccard on q-grams
        sim_cosine_set(to_qgram(a), to_qgram(b))    # Ochiai on q-grams

    val:  str, list, or anything to_list() understands.
    n:    gram size (default 2, i.e. bigrams).
    pad:  optional symbol to pad both ends with (n-1) copies, so edge
          elements appear in as many grams as interior ones. None
          (default) = no padding, the common "shingling" convention.

    Returns a list of n-length grams -- joined strings if `val` is a
    str, else tuples. Sequences shorter than n return an empty list
    (unless padded long enough to reach length n).
    """
    seq = to_list(val)
    if pad is not None:
        seq = [pad] * (n - 1) + seq + [pad] * (n - 1)
    if len(seq) < n:
        return []
    grams = [seq[i:i+n] for i in range(len(seq) - n + 1)]
    if isinstance(val, str):
        return ["".join(str(x) for x in g) for g in grams]
    return [tuple(g) for g in grams]


def to_skipgram(val, n=2, k=1):
    """
    k-skip-n-grams: n-length subsequences of the input where each
    consecutive pair of chosen elements may skip up to k positions
    (i.e. be up to k+1 apart in the original sequence). k=0 reduces to
    ordinary contiguous n-grams -- to_skipgram(val, n, 0) == to_qgram(val, n).

    Like to_qgram, meant to be composed with existing metrics:
        sim_jaccard(to_skipgram(a), to_skipgram(b))    # skip-gram similarity

    Note: this is a brute-force, unoptimized enumeration (fine for
    short educational-scale inputs; combinatorially expensive for long
    sequences or large k -- consistent with simdif's goal of clarity
    over performance).
    """
    seq = to_list(val)
    results = []

    def _extend(start_idx, last_idx, picked):
        if len(picked) == n:
            results.append(tuple(picked))
            return
        limit = len(seq) if last_idx is None else min(len(seq), last_idx + 1 + k + 1)
        for j in range(start_idx, limit):
            _extend(j + 1, j, picked + [seq[j]])

    _extend(0, None, [])
    if isinstance(val, str):
        return ["".join(g) for g in results]
    return results


def to_count_vector(a, b):
    """
    Turn two already-tokenized sequences (e.g. the output of to_qgram
    or to_skipgram, or any list of hashable tokens) into two aligned
    numeric count vectors over their combined vocabulary -- the piece
    Jaccard/cosine_set don't need (they only care about set membership)
    but cosine/Manhattan/Minkowski do (they compare counts position by
    position). Vocabulary order is sorted (by repr, to tolerate mixed
    token types like str grams vs tuple grams) for reproducibility.

    Compose with existing vector metrics:
        va, vb = to_count_vector(to_qgram(a, 2), to_qgram(b, 2))
        sim_cosine(va, vb)         # n-gram cosine
        dist_manhattan(va, vb)     # q-gram distance
    """
    ca, cb = Counter(a), Counter(b)
    vocab = sorted(set(ca) | set(cb), key=repr)
    vec_a = [ca.get(tok, 0) for tok in vocab]
    vec_b = [cb.get(tok, 0) for tok in vocab]
    return vec_a, vec_b

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
    # n_universe=None means "no universe supplied" -> no shared absences (d=0).
    n00 = max(0, (n_universe or 0) - len(a | b))
    return n00, n01, n10, n11


def _align_vectors(a, b, **kwargs):
    pad_value = kwargs.get("pad_value", None)
    if pad_value is not None and len(a) != len(b):
        max_len = max(len(a), len(b))
        a_aligned = list(a) + [pad_value] * (max_len - len(a))
        b_aligned = list(b) + [pad_value] * (max_len - len(b))
        return a_aligned, b_aligned
    return a, b


def to_list_aligned(a, b, **kwargs):
    a = to_list(a, **kwargs)
    b = to_list(b, **kwargs)
    if len(a) != len(b):
        pad_value = kwargs.get('pad_value')
        if pad_value is None:
            raise ValueError("Vector length mismatch")
        a, b = _align_vectors(a, b, pad_value=pad_value)
    return a, b


def to_list_numeric_aligned(a, b, **kwargs):
    a = to_list_numeric(a, **kwargs)
    b = to_list_numeric(b, **kwargs)
    if len(a) != len(b):
        pad_value = kwargs.get('pad_value')
        if pad_value is None:
            raise ValueError("Vector length mismatch")
        numeric_pad = float(pad_value)
        a, b = _align_vectors(a, b, pad_value=numeric_pad)
    return a, b

# ------------------------------------------------------------------
# Edit Distance Metrics
# ------------------------------------------------------------------

_DP_COMBINERS = {"max": max, "min": min, "sum": sum}


def _resolve_cost(param, a, b):
    """A DP cost parameter (insert/delete/substitute/match_score/transpose) is
    either a flat scalar or a callable(a, b) -> cost evaluated against the
    current pair of elements -- e.g. a pointwise distance function for DTW."""
    return param(a, b) if callable(param) else param


def _dp_matrix(s1, s2, insert=1, delete=1, substitute=1, transpose=None, match_score=None, boundary=(1, 1), floor=None, combine="max") -> list:
    rows = len(s1) + 1
    cols = len(s2) + 1
    matrix = [[0] * cols for _ in range(rows)]
    row_step, col_step = boundary
    for i in range(1, rows): matrix[i][0] = matrix[i-1][0] + row_step
    for j in range(1, cols): matrix[0][j] = matrix[0][j-1] + col_step
    combine_fn = combine if callable(combine) else _DP_COMBINERS[combine]
    for i in range(1, rows):
        for j in range(1, cols):
            x, y = s1[i-1], s2[j-1]
            options = [
                matrix[i-1][j] + _resolve_cost(delete, x, y),
                matrix[i][j-1] + _resolve_cost(insert, x, y),
            ]
            diag = matrix[i-1][j-1]
            if x == y:
                # A match is always available (free, or scored by match_score),
                # independent of whether substitution is enabled.
                options.append(diag if match_score is None else diag + _resolve_cost(match_score, x, y))
            elif substitute is not None:
                options.append(diag + _resolve_cost(substitute, x, y))
            if transpose is not None and i > 1 and j > 1:
                if s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    options.append(matrix[i-2][j-2] + _resolve_cost(transpose, x, y))
            cell = combine_fn(options)
            matrix[i][j] = max(floor, cell) if floor is not None else cell
    return matrix

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

def _fill_dp_matrix(a, b, **kwargs):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, **kwargs)
    header_row = [" ", " "] + [str(x) for x in s2]
    side_labels = [" ", " "] + [str(x) for x in s1]
    for i, row in enumerate(matrix):
        row.insert(0, side_labels[i+1])
    matrix.insert(0, header_row)
    return matrix

def _dp_matrix_affine(s1, s2, gap_open=-10, gap_extend=-1, mismatch_penalty=-1,
                       match_score=1, local=False, maximize=True) -> tuple:
    """
    Gotoh's affine-gap DP. Returns (M, Ix, Iy):
        M[i][j]  - best score/dist of an alignment of s1[:i], s2[:j] that
                   ENDS in a match/mismatch (diagonal move)
        Ix[i][j] - best score/dist that ENDS with s1[i-1] aligned to a gap
                   (a "deletion" / gap in s2)
        Iy[i][j] - best score/dist that ENDS with s2[j-1] aligned to a gap
                   (an "insertion" / gap in s1)
    The three are kept separate because whether the *next* gap step is an
    "open" or an "extend" depends on which matrix you were just in -- that's
    the whole reason affine needs three matrices instead of _dp_matrix's one.

    gap_open/gap_extend follow the same sign convention as match_score /
    mismatch_penalty: negative values with maximize=True (alignment score,
    like nw/sw), or positive values with maximize=False (edit distance).
    gap_open is charged once per gap run, gap_extend for each additional
    position in that run (cost of a run of length k = gap_open +
    (k-1)*gap_extend).
    """
    rows, cols = len(s1) + 1, len(s2) + 1
    NEG_INF = -math.inf if maximize else math.inf
    better = max if maximize else min

    M = [[0] * cols for _ in range(rows)]
    Ix = [[0] * cols for _ in range(rows)]
    Iy = [[0] * cols for _ in range(rows)]

    if not local:
        M[0][0] = 0
        for i in range(1, rows):
            M[i][0] = NEG_INF
            Ix[i][0] = gap_open + (i - 1) * gap_extend
            Iy[i][0] = NEG_INF
        for j in range(1, cols):
            M[0][j] = NEG_INF
            Ix[0][j] = NEG_INF
            Iy[0][j] = gap_open + (j - 1) * gap_extend

    for i in range(1, rows):
        for j in range(1, cols):
            sub = match_score if s1[i-1] == s2[j-1] else mismatch_penalty
            diag = better(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1]) + sub
            M[i][j] = max(0, diag) if local else diag

            ix = better(M[i-1][j] + gap_open, Ix[i-1][j] + gap_extend)
            Ix[i][j] = max(0, ix) if local else ix

            iy = better(M[i][j-1] + gap_open, Iy[i][j-1] + gap_extend)
            Iy[i][j] = max(0, iy) if local else iy

    return M, Ix, Iy


def _backtrack_affine(M, Ix, Iy, s1, s2, gap_open, gap_extend, match_score,
                       mismatch_penalty, local=False, maximize=True, gap_symbol="-"):
    rows, cols = len(s1), len(s2)
    better = max if maximize else min

    if local:
        curr_i = curr_j = 0
        best = -math.inf if maximize else math.inf
        for r in range(rows + 1):
            for c in range(cols + 1):
                val = M[r][c]
                if (maximize and val > best) or (not maximize and val < best):
                    best, curr_i, curr_j = val, r, c
        state = 'M'
    else:
        curr_i, curr_j = rows, cols
        scores = {'M': M[curr_i][curr_j], 'Ix': Ix[curr_i][curr_j], 'Iy': Iy[curr_i][curr_j]}
        state = better(scores, key=scores.get)

    align1, align2 = [], []
    while curr_i > 0 or curr_j > 0:
        if local and state == 'M' and M[curr_i][curr_j] == 0:
            break
        if state == 'M':
            sub = match_score if s1[curr_i-1] == s2[curr_j-1] else mismatch_penalty
            align1.append(s1[curr_i-1])
            align2.append(s2[curr_j-1])
            prev = {'M': M[curr_i-1][curr_j-1], 'Ix': Ix[curr_i-1][curr_j-1], 'Iy': Iy[curr_i-1][curr_j-1]}
            state = better(prev, key=prev.get)
            curr_i -= 1
            curr_j -= 1
        elif state == 'Ix':
            align1.append(s1[curr_i-1])
            align2.append(gap_symbol)
            state = 'Ix' if Ix[curr_i][curr_j] == Ix[curr_i-1][curr_j] + gap_extend else 'M'
            curr_i -= 1
        else:  # Iy
            align1.append(gap_symbol)
            align2.append(s2[curr_j-1])
            state = 'Iy' if Iy[curr_i][curr_j] == Iy[curr_i][curr_j-1] + gap_extend else 'M'
            curr_j -= 1

    return align1[::-1], align2[::-1]


def _fill_dp_matrix_affine(a, b, **kwargs):
    """Display helper mirroring _fill_dp_matrix: collapses M/Ix/Iy to a
    single 'best score reaching this cell' grid for human-readable output."""
    s1, s2 = to_list(a), to_list(b)
    maximize = kwargs.get('maximize', True)
    better = max if maximize else min
    M, Ix, Iy = _dp_matrix_affine(s1, s2, **kwargs)
    combined = [[better(M[i][j], Ix[i][j], Iy[i][j]) for j in range(len(s2)+1)] for i in range(len(s1)+1)]
    header_row = [" ", " "] + [str(x) for x in s2]
    side_labels = [" ", " "] + [str(x) for x in s1]
    for i, row in enumerate(combined):
        row.insert(0, side_labels[i+1])
    combined.insert(0, header_row)
    return combined

# ------------------------------------------------------------------
# Metric registry
# ------------------------------------------------------------------
# Populated by the modules in simdif/metrics/, each of which registers its
# own entry (and any aliases) into this dict at import time.

METRICS = {}


# Similarity is the shadow two things cast in the same light. 🦔
