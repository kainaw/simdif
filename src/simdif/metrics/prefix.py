from ..simdif import Metric, METRICS, to_list


def info_prefix() -> str:
    return """
Prefix Similarity
------------------
The length of the longest common PREFIX shared by A and B -- how many
elements match starting from the beginning before the first mismatch.
Cheap to compute (no DP matrix needed) and useful for autocomplete,
sorted-data comparisons, and anywhere "starts the same way" matters more
than overall edit distance.

Formula:
    score(A,B) = length of the longest common prefix of A and B
    sim(A,B)   = score(A,B) / max(|A|, |B|)

Range (score): [0, min(|A|, |B|)]
Range (sim/dif): [0, 1]
    """.strip()


def explain_prefix(a, b, **kwargs) -> str:
    s1, s2 = to_list(a), to_list(b)
    n = score_prefix(a, b, **kwargs)
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Common Prefix: ({", ".join(f"'{x}'" for x in s1[:n])}), length {n}
Similarity: {n} / max({len(s1)}, {len(s2)}) = {sim_prefix(a, b, **kwargs):.4f}
Difference: {dif_prefix(a, b, **kwargs):.4f}
    """.strip()


@Metric
def score_prefix(a, b, **kwargs) -> int:
    s1, s2 = to_list(a), to_list(b)
    n = 0
    for x, y in zip(s1, s2):
        if x != y:
            break
        n += 1
    return n


@Metric
def sim_prefix(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    return score_prefix(s1, s2, **kwargs) / max(len(s1), len(s2))


@Metric
def dif_prefix(a, b, **kwargs) -> float:
    return 1 - sim_prefix(a, b, **kwargs)


METRICS['prefix'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_prefix,
    'sim': sim_prefix,
    'dif': dif_prefix,
    'info': info_prefix,
    'explain': explain_prefix,
}
