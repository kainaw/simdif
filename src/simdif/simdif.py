"""
simdif (and dist, and score, and trace...)
"""

import numbers
import sys

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

VALID_PREFIXES = {'sim', 'dif', 'dist', 'score', 'trace'}

def _resolve_metric(name: str):
    name = name.lower()
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
    role, func, base = _resolve_metric(metric)
    if role != 'sim':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a similarity metric")
    return func(a, b, **kwargs)


def dif(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: dif(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric(metric)
    if role != 'dif':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a difference metric")
    return func(a, b, **kwargs)


def dist(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: dist(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric(metric)
    if role != 'dist':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a distance metric")
    return func(a, b, **kwargs)


def score(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: score(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric(metric)
    if role != 'score':
        raise ValueError(f"Metric '{metric}' is a '{role}', not a scoring metric")
    return func(a, b, **kwargs)


def trace(a, b, metric, **kwargs):
    if isinstance(metric, (list, tuple, set)):
        return {m: trace(a, b, m, **kwargs) for m in metric}
    role, func, base = _resolve_metric(metric)
    if role != 'trace':
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
    if val == None:
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

# ------------------------------------------------------------------
# Set Metrics
# ------------------------------------------------------------------

def sim_jaccard(a, b) -> float:
    """
    Compute the Jaccard similarity between two sets.
    
    The Jaccard index is defined as:
        |A ∩ B| / |A ∪ B|
    
    Parameters
    ----------
    a : iterable or set
        First collection of hashable items.
    b : iterable or set
        Second collection of hashable items.
    
    Returns
    -------
    float
        A value between 0 and 1, where 1 means the sets are identical
        and 0 means they share no elements.
    """
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)

def dif_jaccard(a, b) -> float:
    return 1 - sim_jaccard(a, b)
    
def sim_dice_sorensen(a, b) -> float:
    """
    Compute the Dice similarity coefficient between two sets.
    
    The Dice coefficient is defined as:
        2|A ∩ B| / (|A| + |B|)
    
    Parameters
    ----------
    a : iterable or set
        First collection of items.
    b : iterable or set
        Second collection of items.
    
    Returns
    -------
    float
        A value between 0 and 1 measuring set overlap.
    """
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    intersection = len(a & b)
    return 2 * intersection / (len(a) + len(b))

sim_sorensen_dice = sim_dice_sorensen
sim_dice = sim_dice_sorensen
sim_sorensen = sim_dice_sorensen

def dif_dice_sorensen(a, b) -> float:
    return 1 - sim_dice(a, b)

dif_sorensen_dice = dif_dice_sorensen
dif_dice = dif_dice_sorensen
dif_sorensen = dif_dice_sorensen

def sim_overlap(a, b) -> float:
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    intersection = len(a & b)
    return intersection / min(len(a), len(b))

def dif_overlap(a, b) -> float:
    return 1 - sim_overlap(a, b)

def sim_tversky(a, b, alpha=0.5, beta=0.5) -> float:
    if alpha == 0 and beta == 0:
        raise ValueError("alpha and beta cannot both be 0")
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    intersection = len(a & b)
    return intersection / (intersection + alpha*len(a - b) + beta*len(b - a))

def dif_tversky(a, b, alpha=0.5, beta=0.5) -> float:
    return 1 - sim_tversky(a, b, alpha, beta)

def sim_cosine_set(a, b) -> float:
    a, b = to_set(a), to_set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b)/(len (a) * len(b)) ** 0.5

sim_ochiai = sim_cosine_set

def dif_cosine_set(a, b) -> float:
    return 1 - sim_cosine_set(a, b)

dif_ochiai = dif_cosine_set

# ------------------------------------------------------------------
# Vector Metrics
# ------------------------------------------------------------------

def sim_cosine(a, b) -> float:
    """
    Compute cosine similarity between two numeric vectors.

    Cosine similarity measures the angle between vectors:
        dot(a, b) / (||a|| * ||b||)

    Parameters
    ----------
    a : sequence of numbers
        First numeric vector.
    b : sequence of numbers
        Second numeric vector.

    Returns
    -------
    float
        A value between -1 and 1, where 1 means identical direction,
        0 means orthogonal, and -1 means opposite direction.
    """
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0 and len(b) == 0:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def dif_cosine(a, b) -> float:
    sim = sim_cosine(a, b)
    return -1 - sim if sim < 0 else 1 - sim

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

# ------------------------------------------------------------------
# Edit Distance Metrics
# ------------------------------------------------------------------

def _dp_matrix(s1, s2, match_score, mismatch_penalty, gap_penalty, local, maximize) -> list:
    rows = len(s1) + 1
    cols = len(s2) + 1
    matrix = [[0] * cols for _ in range(rows)]
    
    if not local:  # global alignment (Needleman-Wunsch / Levenshtein)
        for i in range(rows): matrix[i][0] = i * gap_penalty
        for j in range(cols): matrix[0][j] = j * gap_penalty
    
    for i in range(1, rows):
        for j in range(1, cols):
            match = matrix[i-1][j-1] + (match_score if s1[i-1] == s2[j-1] else mismatch_penalty)
            delete = matrix[i-1][j] + gap_penalty
            insert = matrix[i][j-1] + gap_penalty
            cell = max(match, delete, insert) if maximize else min(match, delete, insert)
            matrix[i][j] = max(0, cell) if local else cell
    
    return matrix

def dist_levenshtein(a, b) -> int:
    """
    Compute the Levenshtein edit distance between two sequences.

    The Levenshtein distance is the minimum number of insertions,
    deletions, or substitutions required to transform one sequence
    into the other.

    Parameters
    ----------
    a : sequence
        First sequence (string, list, etc.).
    b : sequence
        Second sequence.

    Returns
    -------
    int
        The edit distance between the two sequences.
    """
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, match_score=0, mismatch_penalty=1, gap_penalty=1, local=False, maximize=False)
    return matrix[-1][-1]

def score_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, maximize=True)
    return matrix[-1][-1]

score_needleman = score_needleman_wunsch
score_wunsch = score_needleman_wunsch

def score_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1) -> int:
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, match_score, mismatch_penalty, gap_penalty, local=True, maximize=True)
    return max(matrix[i][j] for i in range(len(s1)+1) for j in range(len(s2)+1))

score_smith = score_smith_waterman
score_waterman = score_smith_waterman

def sim_levenshtein(a, b) -> float:
    s1 = to_list(a)
    s2 = to_list(b)
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 1.0
    return 1 - (dist_levenshtein(s1, s2) / max_len)

def dif_levenshtein(a, b) -> float:
    return 1 - sim_levenshtein(a, b)

def _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, gap_symbol="-"):
    rows, cols = len(s1), len(s2)
    
    # 1. Determine Starting Point
    if local:
        # Smith-Waterman starts at the highest score in the entire matrix
        curr_i, curr_j = 0, 0
        max_val = -float('inf')
        for r in range(rows + 1):
            for c in range(cols + 1):
                if matrix[r][c] >= max_val:
                    max_val = matrix[r][c]
                    curr_i, curr_j = r, c
    else:
        # Needleman-Wunsch/Levenshtein starts at the bottom-right
        curr_i, curr_j = rows, cols

    align1, align2 = [], []

    # 2. Trace back to the origin (or until 0 for local)
    while curr_i > 0 or curr_j > 0:
        if local and matrix[curr_i][curr_j] == 0:
            break

        current_val = matrix[curr_i][curr_j]

        # Check Diagonal (Match/Mismatch)
        if curr_i > 0 and curr_j > 0:
            score = match_score if s1[curr_i-1] == s2[curr_j-1] else mismatch_penalty
            if current_val == matrix[curr_i-1][curr_j-1] + score:
                align1.append(s1[curr_i-1])
                align2.append(s2[curr_j-1])
                curr_i -= 1
                curr_j -= 1
                continue

        # Check Up (Deletion from s1 / Gap in s2)
        if curr_i > 0 and current_val == matrix[curr_i-1][curr_j] + gap_penalty:
            align1.append(s1[curr_i-1])
            align2.append(gap_symbol)
            curr_i -= 1
        
        # Check Left (Deletion from s2 / Gap in s1)
        else:
            align1.append(gap_symbol)
            align2.append(s2[curr_j-1])
            curr_j -= 1

    # Return as lists (reversed because we traced backwards)
    return align1[::-1], align2[::-1]

def trace_needleman_wunsch(a, b, match_score=1, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, maximize=True)
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=False, gap_symbol=gap_symbol)

def trace_smith_waterman(a, b, match_score=2, mismatch_penalty=-1, gap_penalty=-1, gap_symbol="-"):
    s1, s2 = to_list(a), to_list(b)
    matrix = _dp_matrix(s1, s2, match_score, mismatch_penalty, gap_penalty, local=True, maximize=True)
    return _backtrack(matrix, s1, s2, match_score, mismatch_penalty, gap_penalty, local=True, gap_symbol=gap_symbol)

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

def sim_jarowinkler(a, b, p=0.1, max_l=4) -> float:
    s1, s2 = to_list(a), to_list(b)
    j = sim_jaro(s1, s2)
    l = 0
    for char1, char2 in zip(s1[:max_l], s2[:max_l]):
        if char1 == char2: l += 1
        else: break
    return j + (l * p * (1 - j))

# ------------------------------------------------------------------
# Frequency/Abundance Metrics (for Dicts or Counters)
# ------------------------------------------------------------------

def dist_braycurtis(a, b) -> float:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError("Vector length mismatch")
    diff_sum = sum(abs(x - y) for x, y in zip(v1, v2))
    total_sum = sum(x + y for x, y in zip(v1, v2))
    return diff_sum / total_sum if total_sum != 0 else 0.0

# ------------------------------------------------------------------
# Metric Registry
# ------------------------------------------------------------------

METRICS = {
    'jaccard': {
        'default': 'sim',
        'sim': sim_jaccard,
        'dif': dif_jaccard,
    },
    'dice': {
        'default': 'sim',
        'sim': sim_dice,
        'dif': dif_dice,
    },
    'sorensen': {
        'default': 'sim',
        'sim': sim_sorensen,
        'dif': dif_sorensen,
    },
    'overlap': {
        'default': 'sim',
        'sim': sim_overlap,
        'dif': dif_overlap,
    },
    'tversky': {
        'default': 'sim',
        'sim': sim_tversky,
        'dif': dif_tversky,
    },
    'cosine_set': {
        'default': 'sim',
        'sim': sim_cosine_set,
        'dif': dif_cosine_set,
    },
    'ochiai': {
        'default': 'sim',
        'sim': sim_ochiai,
        'dif': dif_ochiai,
    },

    'cosine': {
        'default': 'sim',
        'sim': sim_cosine,
        'dif': dif_cosine,
    },
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

    'levenshtein': {
        'default': 'dist',
        'dist': dist_levenshtein,
        'sim': sim_levenshtein,
        'dif': dif_levenshtein,
    },

    'needleman_wunsch': {
        'default': 'score',
        'score': score_needleman_wunsch,
        'trace': trace_needleman_wunsch,
    },

    'smith_waterman': {
        'default': 'score',
        'score': score_smith_waterman,
        'trace': trace_smith_waterman,
    },

    'lcs': {
        'default': 'score',
        'score': score_lcs,
        'dist': dist_lcs,
    },

    'jaro': {
        'default': 'sim',
        'sim': sim_jaro,
    },

    'jarowinkler': {
        'default': 'sim',
        'sim': sim_jarowinkler,
    },

    'braycurtis': {
        'default': 'dist',
        'dist': dist_braycurtis,
    },
}


# Similarity is the shadow two things cast in the same light. 🦔