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
            diag = matrix[i-1][j-1]
            if s1[i-1] == s2[j-1]:
                # A match is always available (free, or scored by match_score),
                # independent of whether substitution is enabled.
                options.append(diag if match_score is None else diag + match_score)
            elif substitute is not None:
                options.append(diag + substitute)
            if transpose is not None and i > 1 and j > 1:
                if s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    options.append(matrix[i-2][j-2] + transpose)
            cell = max(options) if maximize else min(options)
            matrix[i][j] = max(0, cell) if local else cell
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

# ------------------------------------------------------------------
# Metric registry
# ------------------------------------------------------------------
# Populated by the modules in simdif/metrics/, each of which registers its
# own entry (and any aliases) into this dict at import time.

METRICS = {}


# Similarity is the shadow two things cast in the same light. 🦔
