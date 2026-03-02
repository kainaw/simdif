from ..simdif import METRICS, _aleph_counts, to_set

def info_rogers_tanimoto() -> str:
    return """
Rogers-Tanimoto Coefficient
---------------------------
Measures similarity by counting both matches (n11 and n00) relative to
the total, but penalizes mismatches by doubling them.
Formula:
    RT(A, B) = (n11 + n00) / (n11 + 2(n10 + n01) + n00)
    Where N = n11 + n10 + n01 + n00
Range: [0, 1]
Aliases: Sokal II, Sokal-Michener II, Sokal-Sneath II
    """.strip()
info_sokal_ii = info_rogers_tanimoto
info_sokal_michener_ii = info_rogers_tanimoto
info_sokal_sneath_ii = info_rogers_tanimoto

def explain_rogers_tanimoto(a, b, n_universe=None, **_) -> str:
    a_set, b_set = to_set(a), to_set(b)
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    n_total = n11 + n10 + n01 + n00
    denominator = n11 + 2 * (n10 + n01) + n00
    intersection = sorted(map(str, a_set & b_set))
    sim = (n11 + n00) / denominator if denominator > 0 else 1.0
    return f"""
A: ({", ".join(sorted(map(str, a_set)))})
B: ({", ".join(sorted(map(str, b_set)))})
Rogers-Tanimoto (Universal Context):
Intersection (n11): {n11} ({", ".join(intersection)})
Total Universe Size (N): {n_total} {f'(Corrected from {n_universe})' if n_universe is not None and n_total != n_universe else ''}
Note: Rogers-Tanimoto requires the total attribute space.
Calculation:
Similarity: ({n11} + {n00}) / ({n11} + 2*({n10} + {n01}) + {n00}) = {sim:.4f}
Difference: 1 - Sim = {1 - sim:.4f}
    """.strip()
explain_sokal_ii = explain_rogers_tanimoto
explain_sokal_michener_ii = explain_rogers_tanimoto
explain_sokal_sneath_ii = explain_rogers_tanimoto

def sim_rogers_tanimoto(a, b, n_universe=None, **_) -> float:
    n00, n01, n10, n11 = _aleph_counts(a, b, n_universe)
    denominator = n11 + 2 * (n10 + n01) + n00
    if denominator == 0:
        return 1.0
    return (n11 + n00) / denominator
sim_sokal_ii = sim_rogers_tanimoto
sim_sokal_michener_ii = sim_rogers_tanimoto
sim_sokal_sneath_ii = sim_rogers_tanimoto

def dif_rogers_tanimoto(a, b, n_universe=None, **_) -> float:
    return 1 - sim_rogers_tanimoto(a, b, n_universe)
dif_sokal_ii = dif_rogers_tanimoto
dif_sokal_michener_ii = dif_rogers_tanimoto
dif_sokal_sneath_ii = dif_rogers_tanimoto

METRICS['rogers_tanimoto'] = {
    'class': 'set',
    'default': 'sim',
    'sim': sim_rogers_tanimoto,
    'dif': dif_rogers_tanimoto,
    'info': info_rogers_tanimoto,
    'explain': explain_rogers_tanimoto,
}
METRICS['sokal_ii'] = METRICS['rogers_tanimoto']
METRICS['sokal_michener_ii'] = METRICS['rogers_tanimoto']
METRICS['sokal_sneath_ii'] = METRICS['rogers_tanimoto']
