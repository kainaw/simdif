"""Internal helpers shared by sibling metric modules.

These are implementation details, not metrics - nothing here registers into
METRICS or is re-exported by metrics/__init__.py (every name starts with '_').
Each helper lives here only because more than one metric file needs it; helpers
used by a single metric stay in that metric's own module.
"""
from ..simdif import to_list, to_list_aligned


# --- Two-sample statistics (welch_t, cohens_d) --------------------------------

def _mean_var(x):
    """(n, mean, sample_variance) for a numeric sample. Variance uses n-1."""
    n = len(x)
    if n < 2:
        raise ValueError("Each sample needs at least 2 values to estimate variance")
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / (n - 1)
    return n, mean, var


# --- Clustering / partition comparison (rand_index, adjusted_rand, fowlkes) ----

def _labels(a, b):
    la, lb = to_list(a), to_list(b)
    if len(la) != len(lb):
        raise ValueError("Vector length mismatch")
    return la, lb


def _pair_counts(la, lb):
    """Over all object pairs, count agreements/disagreements between two
    clusterings given as aligned label sequences:
        a = together in both        b = together in A only
        c = together in B only      d = apart in both
    Returns (a, b, c, d). Label-invariant: only '==' of labels is used."""
    n = len(la)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = la[i] == la[j]
            same_b = lb[i] == lb[j]
            if same_a and same_b:
                a += 1
            elif same_a:
                b += 1
            elif same_b:
                c += 1
            else:
                d += 1
    return a, b, c, d


def _pair_table(a, b, c, d):
    total = a + b + c + d
    return f"""Object pairs (C(n,2) = {total}):
  a  together in both   = {a}   (agreement)
  b  together in A only = {b}   (disagreement)
  c  together in B only = {c}   (disagreement)
  d  apart in both      = {d}   (agreement)"""


# --- Evolutionary distances (p_distance, jukes_cantor) ------------------------

def _seq_diffs(a, b, **kwargs):
    """Two aligned sequences of any '=='-comparable elements -> (n_sites, n_diff).

    Generic: only equality is used, so it applies to DNA, protein, words, or
    arbitrary tokens. Raises 'Vector length mismatch' unless a pad_value is
    supplied (same contract as hamming).
    """
    a, b = to_list_aligned(a, b, **kwargs)
    return len(a), sum(x != y for x, y in zip(a, b))
