from ..simdif import Metric, METRICS, to_list


def info_suffix() -> str:
    return """
Suffix Similarity
------------------
The length of the longest common SUFFIX shared by A and B -- how many
elements match counting backward from the end before the first mismatch.
Useful for morphological comparisons (shared word endings), file
extensions, and anywhere "ends the same way" matters more than overall
edit distance.

Formula:
    score(A,B) = length of the longest common suffix of A and B
    sim(A,B)   = score(A,B) / max(|A|, |B|)

Range (score): [0, min(|A|, |B|)]
Range (sim/dif): [0, 1]
    """.strip()


def explain_suffix(a, b, **kwargs) -> str:
    s1, s2 = to_list(a), to_list(b)
    n = score_suffix(a, b, **kwargs)
    tail = s1[len(s1)-n:] if n else []
    return f"""
A: ({", ".join(f"'{x}'" for x in s1)})
B: ({", ".join(f"'{y}'" for y in s2)})
Common Suffix: ({", ".join(f"'{x}'" for x in tail)}), length {n}
Similarity: {n} / max({len(s1)}, {len(s2)}) = {sim_suffix(a, b, **kwargs):.4f}
Difference: {dif_suffix(a, b, **kwargs):.4f}
    """.strip()


@Metric
def score_suffix(a, b, **kwargs) -> int:
    s1, s2 = to_list(a), to_list(b)
    n = 0
    for x, y in zip(reversed(s1), reversed(s2)):
        if x != y:
            break
        n += 1
    return n


@Metric
def sim_suffix(a, b, **kwargs) -> float:
    s1, s2 = to_list(a), to_list(b)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    return score_suffix(s1, s2, **kwargs) / max(len(s1), len(s2))


@Metric
def dif_suffix(a, b, **kwargs) -> float:
    return 1 - sim_suffix(a, b, **kwargs)


METRICS['suffix'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_suffix,
    'sim': sim_suffix,
    'dif': dif_suffix,
    'info': info_suffix,
    'explain': explain_suffix,
}
