from ..simdif import Metric, METRICS, sim, to_tokens


def info_monge_elkan() -> str:
    return """
Monge-Elkan Similarity
----------------------
A token-level similarity for multi-word strings. Each token of A is matched to
its best-scoring token in B using an inner secondary similarity (Jaro-Winkler by
default), and the results are averaged over the tokens of A.

Formula:
    ME(A, B) = (1 / |A|) * sum_{a in A} max_{b in B} sim(a, b)

Range: depends on the inner similarity (typically [0, 1])

Note: This metric is asymmetric — ME(A, B) may differ from ME(B, A). The inner
metric can be selected with the `method` argument (any similarity metric name).
    """.strip()


def explain_monge_elkan(a, b, method="jaro_winkler", **kwargs) -> str:
    tokens_a = to_tokens(a)
    tokens_b = to_tokens(b)
    if not tokens_a or not tokens_b:
        return "One side has no tokens; Monge-Elkan = 0.0"
    lines = []
    for s in tokens_a:
        best_t = max(tokens_b, key=lambda t: sim(s, t, method))
        lines.append(f"  '{s}' -> best '{best_t}' = {sim(s, best_t, method):.4f}")
    return f"""
A tokens: {tokens_a}
B tokens: {tokens_b}
Inner similarity: {method}
Best match per token of A:
{chr(10).join(lines)}
Monge-Elkan (average): {sim_monge_elkan(a, b, method):.4f}
    """.strip()


@Metric
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


METRICS['monge_elkan'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_monge_elkan,
    'info': info_monge_elkan,
    'explain': explain_monge_elkan,
}
