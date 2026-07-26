from ..simdif import Metric, METRICS, sim, dist, dif, score, to_tokens

_ROLE_DISPATCH = {'sim': sim, 'dist': dist, 'dif': dif, 'score': score}
_ROLE_PICK = {'sim': max, 'dist': min, 'dif': min, 'score': max}
_COMBINE = {
    'avg': lambda xs: sum(xs) / len(xs),
    'sum': sum,
    'min': min,
    'max': max,
}


def info_monge_elkan() -> str:
    return """
Monge-Elkan
-----------
A token-level metric for multi-word strings. Each token of A is matched to
its best token in B using an inner `method` metric (Levenshtein by default),
then the per-token best scores are reduced to one number via `combine`.

Formula:
    ME(A, B) = combine_{a in A} best_{b in B} method(a, b)

`sim_monge_elkan`/`score_monge_elkan` use the `sim`/`score` role of `method`
and take the best (max) score per token. `dist_monge_elkan`/`dif_monge_elkan`
use the `dist`/`dif` role and take the best (min) score per token. `combine`
reduces the per-token best scores across all of A: "avg" (default), "sum",
"min", or "max".

Range: depends on `method`'s role (typically [0, 1] for sim/dif, unbounded
for dist)

Note: This metric is asymmetric — ME(A, B) may differ from ME(B, A).

Note: `method` must expose the role being requested, or a ValueError is
raised (e.g. `method="cosine"` has no `dist` role, so
`dist_monge_elkan(a, b, method="cosine")` fails). Levenshtein exposes all
three roles, so it always runs — but its `sim`/`dif` roles are just a
normalized edit distance, not a true similarity, so `sim_monge_elkan` with
an edit-distance-style `method` can produce misleading numbers.
`dist_monge_elkan` is the natural fit for those methods.
    """.strip()


def explain_monge_elkan(a, b, method="levenshtein", combine="avg", role="sim") -> str:
    tokens_a = to_tokens(a)
    tokens_b = to_tokens(b)
    if not tokens_a or not tokens_b:
        return "One side has no tokens; Monge-Elkan = 0.0"
    dispatch = _ROLE_DISPATCH[role]
    pick = _ROLE_PICK[role]
    lines = []
    picks = []
    for s in tokens_a:
        scores = {t: dispatch(s, t, method) for t in tokens_b}
        best_t = pick(scores, key=scores.get)
        picks.append(scores[best_t])
        lines.append(f"  '{s}' -> best '{best_t}' ({role} {method}) = {scores[best_t]:.4f}")
    result = _COMBINE[combine](picks)
    return f"""
A tokens: {tokens_a}
B tokens: {tokens_b}
Role: {role}  Method: {method}  Combine: {combine}
Best match per token of A:
{chr(10).join(lines)}
Per-token best scores: {[round(p, 4) for p in picks]}
Monge-Elkan ({combine}): {result:.4f}
    """.strip()


def _monge_elkan(a, b, method, combine, role) -> float:
    tokens_a = to_tokens(a)
    tokens_b = to_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    dispatch = _ROLE_DISPATCH[role]
    pick = _ROLE_PICK[role]
    picks = [pick(dispatch(s, t, method) for t in tokens_b) for s in tokens_a]
    return _COMBINE[combine](picks)


@Metric
def sim_monge_elkan(a, b, method="levenshtein", combine="avg") -> float:
    return _monge_elkan(a, b, method, combine, role='sim')


@Metric
def dist_monge_elkan(a, b, method="levenshtein", combine="avg") -> float:
    return _monge_elkan(a, b, method, combine, role='dist')


@Metric
def dif_monge_elkan(a, b, method="levenshtein", combine="avg") -> float:
    return _monge_elkan(a, b, method, combine, role='dif')


@Metric
def score_monge_elkan(a, b, method="levenshtein", combine="avg") -> float:
    return _monge_elkan(a, b, method, combine, role='score')


METRICS['monge_elkan'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_monge_elkan,
    'dist': dist_monge_elkan,
    'dif': dif_monge_elkan,
    'score': score_monge_elkan,
    'info': info_monge_elkan,
    'explain': explain_monge_elkan,
}
