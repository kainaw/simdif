# simdif
*A unified similarity, difference, distance, scoring, and alignment library for Python.*

---

## Overview

`simdif` is a single, unified library that brings together:

- **Set similarity metrics**
- **Vector similarity & distance metrics**
- **Sequence alignment algorithms**
- **Edit distance algorithms**
- **Frequency / abundance metrics**
- **Traceback alignment generation**

Most libraries focus on *one* of these.  
`simdif` treats them all as **expressions of the same core idea**:

> How similar, different, or distant are two things?

This allows you to switch metrics without changing your mental model.

---

## Core Concepts

`simdif` organizes metrics into **five fundamental categories**:

| Prefix | Meaning | Output Range | Examples |
|---------|----------|---------------|------------|
| `sim_`  | Similarity | Usually `[0, 1]` | Jaccard, Cosine, Dice |
| `dif_`  | Difference | Usually `[0, 1]` | Levenshtein, Hamming |
| `dist_` | Distance | `[0, ∞)` | Euclidean, Manhattan, Edit Distance |
| `score_` | Alignment Score | Unbounded | Needleman–Wunsch, Smith–Waterman |
| `trace_` | Alignment Trace | Aligned sequences | Global & Local alignment |

Every metric belongs to **exactly one conceptual role**.

---

## Unified Dispatch API

All metrics can be accessed dynamically using:

```python
sim(a, b, "metric")
dif(a, b, "metric")
dist(a, b, "metric")
score(a, b, "metric")
trace(a, b, "metric")
````

A common simdif function can be used in place of the independent functions:

```python
simdiff(a, b, "metric")
```

The most common form of the metric will be selected.
For example, "jaccard" is commonly used as "sim_jaccard" instead of "dif_jaccard".
You can force the form by using a full metric name, such as "dif_jaccard".

### Example

```python
from simdif import sim, dist

sim("night", "nacht", "jaro")
dist([1,2,3], [4,5,6], "euclidean")
```

This lets you **swap metrics without changing your code structure**.

If you sent more than one metric in a list structure, the function will return a dict of results.
Example:

```python
similarities = sim("hedge", "hog", ['jaccard','dice','jaro'])
# returns: {'jaccard':0.4, 'dice':0.5714, 'jaro':0.6889}
```

---

## Supported Metrics

### Set Metrics

| Metric                  | Description                   |
| ----------------------- | ----------------------------- |
| `jaccard`               | Intersection over union       |
| `dice` / `sorensen`     | Twice intersection over total |
| `overlap`               | Intersection over smaller set |
| `tversky`               | Weighted generalization       |
| `cosine_set` / `ochiai` | Cosine similarity for sets    |

---

### Vector Metrics

| Metric     | Description             |
| ---------- | ----------------------- |
| `cosine`   | Vector angle similarity |
| `hamming`  | Element-wise mismatch   |
| `tanimoto` | Generalized Jaccard     |

---

### Distance Metrics

| Metric      | Description    |
| ----------- | -------------- |
| `minkowski` | General p-norm |
| `euclidean` | L2 norm        |
| `manhattan` | L1 norm        |
| `chebyshev` | L∞ norm        |

---

### Edit Distance & Alignment

| Metric             | Description                |
| ------------------ | -------------------------- |
| `levenshtein`      | Minimum edit distance      |
| `needleman_wunsch` | Global alignment           |
| `smith_waterman`   | Local alignment            |
| `lcs`              | Longest common subsequence |
| `jaro`             | Jaro string similarity     |
| `jarowinkler`      | Prefix-weighted Jaro       |

---

### Abundance Metrics

| Metric       | Description                         |
| ------------ | ----------------------------------- |
| `braycurtis` | Abundance-based ecological distance |

---

## Traceback Alignment

`trace_*` functions return **actual sequence alignments**.

```python
from simdif import trace

a, b = trace("GATTACA", "GCATGCU", "needleman_wunsch")

print("".join(a))
print("".join(b))
```

Output:

```
G-ATTACA
GCA-TGCU
```

---

## Input Flexibility

`simdif` accepts:

* strings
* lists
* tuples
* sets
* dicts
* numpy arrays
* pandas Series
* torch tensors

All are automatically normalized internally.

---

## Design Philosophy

Most similarity libraries suffer from **narrow thinking**:

* NLP libraries → string edit distance
* ML libraries → vector similarity
* Graph libraries → structural distance

`simdif` takes a **unifying approach**:

> A string is a sequence.
> A vector is a sequence.
> A set is a degenerate sequence.

Everything becomes a structured comparison problem.

---

## Metric Taxonomy

```
Similarity
├── Set
│   ├── Jaccard
│   ├── Dice / Sørensen
│   ├── Overlap
│   └── Tversky
│
├── Vector
│   ├── Cosine
│   └── Tanimoto
│
└── Sequence
    ├── Jaro
    └── Jaro-Winkler


Difference
├── Levenshtein
├── Hamming
└── Overlap variants


Distance
├── Minkowski (p)
│   ├── Euclidean (p=2)
│   ├── Manhattan (p=1)
│   └── Chebyshev (p=∞)
└── Bray-Curtis


Alignment
├── Needleman-Wunsch (global)
└── Smith-Waterman (local)
```

---

## Why Another Similarity Library?

Because **existing libraries fragment the concept**.

| Library            | Limitation             |
| ------------------ | ---------------------- |
| python-Levenshtein | strings only           |
| scipy.spatial      | vectors only           |
| jellyfish          | string heuristics only |
| nltk               | NLP-specific           |

`simdif` unifies **sets, vectors, strings, sequences, and alignments** under one coherent API.

---

## Installation

```bash
pip install simdif
```

*(coming soon)*

---

## License

MIT

---

> **Similarity is the shadow two things cast in the same light. 🦔**
