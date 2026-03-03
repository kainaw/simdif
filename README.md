# simdif

**simdif** is a pure-Python library for computing, comparing, and understanding similarity, difference, and distance metrics. It started as a collection of similarity and difference metrics, and has grown to include distance metrics and alignment scores (Smith-Waterman, Needleman-Wunsch). The design goal is **education first**: every metric ships with a step-by-step explanation function and a plain-English definition function so you can see exactly how a score is derived.

> ⚠️ **Not intended for large-scale or production workloads.** simdif is pure Python and prioritises clarity over speed. It is well-suited for classroom use, small experiments, and learning how metrics work — not for processing large datasets or performance-critical pipelines.

---

## Highlights

- **50+ metrics** under one unified interface
- **Three input classes**: sets, sequences, and vectors
- **Broad input support**: plain Python lists, strings, sets, numeric lists, and NumPy / SciPy / TensorFlow arrays/tensors
- **Compare metrics side-by-side** by passing a list of metric names to `simdif()`
- **`explain_<metric>()`** — walks through the calculation step by step and returns the score
- **`info_<metric>()`** — returns a plain-English definition of the metric
- **Alias-friendly**: most metrics have multiple accepted names (e.g. `'dice'`, `'sorensen'`, `'dice_sorensen'`, `'sorensen_dice'` all work)
- **Optional fast paths**: if a faster library (e.g. `python-Levenshtein`) is installed, simdif will lazy-import and use it automatically — no configuration needed

---

## Installation

```bash
# Clone and install in editable mode (development)
git clone https://github.com/kainaw/simdif.git
cd simdif
pip install -e .
```

> Optional speedup libraries (simdif will use them automatically if present):
> ```bash
> pip install python-Levenshtein
> ```

---

## Quickstart

### Compare multiple metrics at once

```python
from simdif import simdif

result = simdif(
    "Hedge", "Hog",
    ['jaccard', 'dice', 'cosine', 'levenshtein', 'pearson'],
    ascii=True,
    pad_value='0'
)
print(result)
```

`simdif()` returns a single dictionary keyed by metric name so you can compare results directly:

```python
{
    'jaccard':      0.375,
    'dice':         0.545,
    'cosine':       0.756,
    'levenshtein':  3,
    'pearson':      0.823
}
```

### Understand a metric step by step

```python
from simdif import explain_jaccard

explain_jaccard("Hedge", "Hog", ascii=True, pad_value='0')
```

```
Input A (as set): {'H', 'e', 'd', 'g'}
Input B (as set): {'H', 'o', 'g'}

Intersection |A ∩ B|: {'H', 'g'} → 2
Union        |A ∪ B|: {'H', 'e', 'd', 'g', 'o'} → 5

Jaccard = |A ∩ B| / |A ∪ B| = 2 / 5 = 0.4
```

### Look up a metric definition

```python
from simdif import info_jaccard

info_jaccard()
```

```
Jaccard Similarity (Jaccard Index / IoU)
----------------------------------------
Measures the overlap between two sets as a fraction of their union.
Range: 0 (no overlap) to 1 (identical sets).
Common uses: document similarity, recommendation systems, clustering evaluation.
```

---

## Input Classes

simdif organises metrics into three classes based on what kind of input they operate on. Passing the wrong type of input for a metric will raise an informative error.

| Class | What it operates on | Example inputs |
|---|---|---|
| **Set** | Unordered collections; duplicates ignored | `{'a','b','c'}`, string→set, list→set |
| **Sequence** | Ordered elements; position matters | strings, lists, tuples |
| **Vector** | Numeric arrays; magnitude and direction matter | lists of numbers, NumPy arrays, SciPy sparse, TensorFlow tensors |

### String / ASCII helpers

When comparing strings as sets or vectors, use `ascii=True` with a `pad_value` to control how characters are encoded and how unequal-length inputs are handled:

```python
simdif("cat", "cart", ['jaccard', 'cosine'], ascii=True, pad_value='0')
```

---

## Metrics Reference

Metrics marked with an alias share their implementation with the canonical name. All aliases are fully supported in both `simdif()` calls and standalone `explain_` / `info_` functions.

### Set Metrics

These metrics treat inputs as unordered collections. Element frequency is ignored; only membership matters.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `cosine_set` | `ochiai` | sim |
| `dice_sorensen` | `dice`, `sorensen`, `sorensen_dice` | sim |
| `jaccard` | `iou` | sim |
| `kulczynski` | `kulczynski_ii` | sim |
| `overlap` | `szymkiewicz_simpson`, `simpson` | sim |
| `rogers_tanimoto` | `sokal_ii`, `sokal_michener_ii`, `sokal_sneath_ii` | sim |
| `russel_rao` | `russell_rao`, `rr` | sim |
| `smc` | `sokal_michener` | sim |
| `sokal_sneath_i` | `ssi` | sim |
| `sokal_sneath_iii` | `ssiii` | sim |
| `tversky` | — | sim |

### Sequence Metrics

These metrics treat inputs as ordered. The position of elements matters (e.g. `"abc" ≠ "bca"`).

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `damerau_levenshtein` | `dl` | dist |
| `hamming` | — | dist |
| `jaro` | — | sim |
| `jaro_winkler` | — | sim |
| `kendall_tau` | `kendall_tau_a`, `tau_a` | sim |
| `kendall_tau_b` | `tau_b` | sim |
| `levenshtein` | — | dist |
| `lcs` | — | score |
| `monge_elkan` | — | sim |
| `needleman_wunsch` | — | score |
| `osa` | — | dist |
| `smith_waterman` | — | score |
| `soundex` | — | sim |
| `spearman` | — | sim |

### Vector Metrics

These metrics operate on numeric arrays. Inputs must be numeric and equal in length (or padded).

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `bray_curtis` | — | dist |
| `canberra` | — | dist |
| `chebyshev` | `chessboard`, `linf` | dist |
| `cosine` | — | sim |
| `euclidean` | — | dist |
| `hedgehog` | — | sim |
| `index_of_dissimilarity` | `hoover`, `duncan` | dif |
| `mahalanobis` | — | dist |
| `manhattan` | `taxicab`, `cityblock` | dist |
| `minkowski` | — | dist |
| `pearson` | — | sim |

### Probabilistic / Divergence Metrics

These metrics measure how much two probability distributions differ.

| Canonical Name | Aliases | Default Output |
|---|---|---|
| `kl_divergence` | `kullback_leibler` | dist |
| `js_divergence` | `jensen_shannon` | dist |
| `tanimoto` | — | sim |

---

## Output Types

Many metrics can return more than one type of output. The table below shows what each output type means:

| Output type | Meaning |
|---|---|
| `sim` | Similarity — higher is more similar (typically 0–1) |
| `dif` | Difference — higher is more different (typically 0–1) |
| `dist` | Distance — higher means further apart (range varies by metric) |
| `score` | Raw alignment score (Smith-Waterman, Needleman-Wunsch, LCS) |
| `matrix` | Full dynamic programming matrix (Levenshtein, NW, SW, LCS) |
| `trace` | Alignment traceback path (NW, SW) |

To request a specific output type:

```python
from simdif import simdif

# Get distance instead of the default similarity
simdif("cat", "car", ['levenshtein'], output='dist')

# Get the full DP matrix
simdif("cat", "car", ['levenshtein'], output='matrix')
```

---

## The Educational Interface

Every metric in simdif that has a `class` designation (`set`, `sequence`, or `vector`) exposes two educational functions.

### `explain_<metric>(a, b, ...)`

Runs the calculation and prints each step — what the inputs look like after preprocessing, what intermediate values are computed, and how the final score is assembled. Returns the score so it can be used programmatically.

### `info_<metric>()`

Prints a human-readable description of the metric: what it measures, its output range, and typical use cases. Takes no arguments.

```python
from simdif import explain_cosine, info_cosine

info_cosine()        # What is cosine similarity?
explain_cosine([1, 2, 3], [4, 5, 6])  # Show me the dot products, magnitudes, etc.
```

---

## Comparing Metrics Side by Side

One of simdif's most useful features for education is running many metrics over the same pair of inputs to see how they agree or disagree:

```python
from simdif import simdif

a = [1, 0, 1, 1, 0]
b = [1, 1, 0, 1, 0]

results = simdif(a, b, [
    'jaccard', 'dice', 'cosine_set',          # set metrics
    'hamming', 'kendall_tau',                  # sequence metrics
    'cosine', 'euclidean', 'manhattan'         # vector metrics
])

for name, score in results.items():
    print(f"{name:>20}: {score:.4f}")
```

This is especially useful for showing students that the "right" metric depends on what you care about — membership, order, magnitude, or distribution.

---

## Known Limitations

- **Not optimised for performance.** Pure Python implementations mean simdif is slow on large inputs. It is not a replacement for NumPy, SciPy, or specialised libraries like `rapidfuzz`.
- **Explanation output can be verbose on long inputs.** `explain_*` functions are designed for short, illustrative examples.
- **Some metrics require specific input types.** Passing a non-numeric input to a vector metric, or a non-sequence to a sequence metric, will raise an error.
- **Work in progress.** The library is functional but not fully polished. APIs may change.

---

## Contributing

Contributions, bug reports, and suggestions are welcome. If you add a new metric, please include both an `info_` and `explain_` function to keep the educational interface consistent.

---

## License

MIT
