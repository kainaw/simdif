from ..simdif import Metric, METRICS, _aleph_counts, to_set


def info_sokal_sneath_iii() -> str:
    return """
Sokal-Sneath III Similarity Coefficient (SS3)
---------------------------------------------
A non-normalized ratio metric that compares the total number of matches 
(both positive and negative) against the total number of mismatches.

Formula:
    SS3(A, B) = (n11 + n00) / (n10 + n01)
    Where:
        n11 = Positive matches
        n00 = Negative matches (shared absences)
        n10 + n01 = Total mismatches

Range: [0, inf]
    inf = Identical (zero mismatches)
    1   = Matches and mismatches are equal
    0   = No matches at all

Note: This is a ratio, not a probability. It is not bounded by 1.0.
    """.strip()
info_ssiii = info_sokal_sneath_iii


def explain_sokal_sneath_iii(a, b, **_) -> str:
    a, b = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b)
    matches = n11 + n00
    mismatches = n10 + n01
    similarity = matches / mismatches if mismatches > 0 else float('inf')
    return f"""
A: {sorted(list(a))}, length {len(a)}
B: {sorted(list(b))}, length {len(b)}
Total Matches (n11 + n00): {matches}
Total Mismatches (n10 + n01): {mismatches}
Similarity Ratio: {matches} / {mismatches} = {similarity}
    """.strip()
explain_ssiii = explain_sokal_sneath_iii


@Metric
def sim_sokal_sneath_iii(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    mismatches = n10 + n01
    if mismatches == 0:
        return float('inf')
    return (n11 + n00) / mismatches
sim_ssiii = sim_sokal_sneath_iii


@Metric
def dif_sokal_sneath_iii(a, b, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b)
    matches = n11 + n00
    if matches == 0:
        return float('inf')
    return (n10 + n01) / matches
dif_ssiii = dif_sokal_sneath_iii


METRICS['sokal_sneath_iii'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_sokal_sneath_iii,
    'dif': dif_sokal_sneath_iii,
    'info': info_sokal_sneath_iii,
    'explain': explain_sokal_sneath_iii,
}
METRICS['ssiii'] = METRICS['sokal_sneath_iii']
