from ..simdif import METRICS, _aleph_counts, to_set

def info_russel_rao() -> str:
    return """
Russel-Rao Coefficient
----------------------
Measures similarity as the intersection (matches) relative to the total
attribute universe (N), including the "null-null" cases where neither set
possesses an attribute.

Formula:
    RR(A, B) = |A ∩ B| / N
    Where N = n11 + n10 + n01 + n00

Range: [0, 1]

Aliases: Russell-Rao (common misspelling), RR
    """.strip()
info_russell_rao = info_russel_rao
info_rr = info_russel_rao

def explain_russel_rao(a, b, n_universe=None, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00

    intersection = sorted(map(str, a_set & b_set))
    sim = n11 / n_total if n_total > 0 else 1.0

    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})

Russel-Rao (Universal Context):
Intersection (n11): {n11} ({", ".join(intersection)})
Total Universe Size (N): {n_total} {f'(Corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''}

Note: Russel-Rao requires the total attribute space.

Calculation:
Similarity: {n11} / {n_total} = {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()
explain_russell_rao = explain_russel_rao
explain_rr = explain_russel_rao

def sim_russel_rao(a, b, n_universe=None, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    if (n11 + n10 + n01 + n00) == 0:
        return 1.0
    return n11 / (n11 + n10 + n01 + n00)
sim_russell_rao = sim_russel_rao
sim_rr = sim_russel_rao


def dif_russel_rao(a, b, n_universe=None, **_) -> float:
    return 1 - sim_russel_rao(a, b, n_universe)
dif_russell_rao = dif_russel_rao
dif_rr = dif_russel_rao


METRICS['russel_rao'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_russel_rao,
    'dif': dif_russel_rao,
    'info': info_russel_rao,
    'explain': explain_russel_rao,
}
METRICS['russell_rao'] = METRICS['russel_rao']
METRICS['rr'] = METRICS['russell_rao']
