"""Internal helpers shared by sibling metric modules.

These are implementation details, not metrics - nothing here registers into
METRICS or is re-exported by metrics/__init__.py (every name starts with '_').
Each helper lives here only because more than one metric file needs it; helpers
used by a single metric stay in that metric's own module.
"""
from ..simdif import to_list, to_list_aligned


# --- Bounding an unbounded distance (canberra, js_divergence, lee, geodesic) ---

def _bounded_dif(d, d_max):
    """Map a distance onto [0, 1] by dividing by its maximum: dif = d / d_max.

    Only for distances whose maximum is genuinely known -- either a constant
    (js_divergence: ln 2) or a closed form in the inputs (canberra: n, lee:
    floor(q/2)*n, geodesic: pi*radius). The companion similarity is 1 - dif,
    so sim + dif == 1 exactly. Metrics with no derivable maximum use the
    1/(1+d) squash instead and cannot use this.

    d_max <= 0 means there was nothing to measure (two empty inputs, where
    d = 0 as well), so the pair is identical and the difference is 0.

    The clamp exists to absorb floating-point overshoot, not to rescue a wrong
    bound: clamping a real overshoot would flatten every distance above d_max
    onto an identical 1.0, destroying the ordering the metric was computed for.
    An infinite d clamps to 1.0, which is the right answer for a pair that is
    maximally far apart; a nan propagates.
    """
    if d_max <= 0:
        return 0.0
    return min(d / d_max, 1.0)


# --- Bounding a distance with NO derivable maximum ----------------------------
#
# The Minkowski family and friends have no inherent ceiling: on R^n there is no
# such thing as "maximally far apart", so there is nothing to divide by. Two
# branches, and which role is computed first differs between them:
#
#   no d_max -> sim = 1 / (1 + d) is primitive, dif = 1 - sim
#   d_max    -> dif = d / d_max   is primitive, sim = 1 - dif
#
# sim + dif == 1 either way. The squash never reaches 0 or 1 for finite d, so
# "completely different" is unreachable without a bound -- which is honest,
# because without a bound it is undefined.
#
# d_max is read from kwargs rather than being a named parameter, so passing it
# alongside a list of metrics is harmless: metrics with a derivable maximum
# (canberra, lee, geodesic, ...) and metrics with a fixed range simply ignore
# it, exactly as they already ignore pad_value, q, and radius. Do not promote it
# to a named parameter -- several of these modules call the builtin max() in
# their own bodies.

def _sim_from_dist(d, d_max=None):
    """Bounded similarity for a distance with no derivable maximum.

    Without d_max this is the 1/(1+d) squash, which is the primitive in that
    branch: computing it directly keeps precision for large d, where the
    difference has saturated to 1.0 and can no longer tell pairs apart. An
    infinite d gives 0.0 rather than nan.
    """
    if d_max is None:
        return 1.0 / (1.0 + d)
    return 1.0 - _bounded_dif(d, d_max)


def _dif_from_dist(d, d_max=None):
    """Bounded difference for a distance with no derivable maximum.

    With d_max this is the primitive, d / d_max. Without one it is derived as
    1 - sim, which costs a little precision for very small d (where sim has
    rounded to 1.0) but keeps sim + dif == 1 exact and maps an infinite d to
    1.0 instead of nan.
    """
    if d_max is None:
        return 1.0 - (1.0 / (1.0 + d))
    return _bounded_dif(d, d_max)


def _max_line(d, d_max, unbounded_note="unbounded -- no maximum exists"):
    """The explain_ lines reporting how a distance was bounded.

    Mirrors the '(derived)' line the known-maximum metrics print, so the two
    branches read the same way and a d_max that was silently ignored by the
    wrong metric is visible as '(derived)' instead of '(supplied)'.
    """
    if d_max is None:
        return (f"Maximum: {unbounded_note}\n"
                f"Similarity (1 / (1 + d)): {_sim_from_dist(d):.4f}\n"
                f"Difference (1 - sim): {_dif_from_dist(d):.4f}\n"
                f"Pass d_max to rescale these linearly against a known bound.")
    dif = _dif_from_dist(d, d_max)
    clamp = f" (clamped from {d / d_max:.4f})" if d > d_max else ""
    return (f"Maximum: d_max = {d_max:.4f} (supplied)\n"
            f"Difference (dist / d_max): {d:.4f} / {d_max:.4f} = {dif:.4f}{clamp}\n"
            f"Similarity (1 - dif): {1.0 - dif:.4f}")


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
