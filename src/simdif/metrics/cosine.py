from ..simdif import Metric, METRICS, to_list_numeric
import sys


def info_cosine() -> str:
    return """
Cosine Similarity
-----------------
Measures the angle between two numeric vectors, ignoring magnitude.
Two vectors pointing in the same direction score 1.0 regardless of
their lengths. Widely used in NLP, recommendation systems, and any
domain where direction matters more than scale.
Formula:
    cos(A,B) = dot(A,B) / (||A|| * ||B||)
             = Σ(a_i * b_i) / (√Σa_i² * √Σb_i²)
Range: [-1, 1]
    1  = identical direction
    0  = orthogonal (no correlation)
   -1  = opposite direction
Difference: [0, 2]  — normalized to [0, 1] for negative similarities
Distance:   [0, 2]  — 1 - similarity (standard cosine distance)
Note: For binary/set data, use cosine_set instead, which is equivalent
to the Ochiai coefficient.
    """.strip()


def explain_cosine(a, b, **_) -> str:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    sim = sim_cosine(a, b)
    return f"""
A: ({", ".join(map(str, a))})
B: ({", ".join(map(str, b))})
Cosine Similarity:
dot(A,B):  {dot}
||A||:     {norm_a:.4f}
||B||:     {norm_b:.4f}
||A||*||B||: {norm_a * norm_b:.4f}
Calculation:
  dot(A,B) / (||A|| * ||B||)
= {dot} / {norm_a * norm_b:.4f}
= {sim:.4f}
Difference: {dif_cosine(a, b):.4f}
Distance:   {dist_cosine(a, b):.4f}
    """.strip()


@Metric
def sim_cosine(a, b, **_) -> float:
    a, b = to_list_numeric(a), to_list_numeric(b)
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if 'scipy' in sys.modules:
        from scipy.spatial import distance
        return 1.0 - float(distance.cosine(a, b))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


@Metric
def dist_cosine(a, b, **_) -> float:
    return 1 - sim_cosine(a, b)


METRICS['cosine'] = {
    'class': 'vector',
    'default': 'sim',
    'sim': sim_cosine,
    'dist': dist_cosine,
    'info': info_cosine,
    'explain': explain_cosine,
}
