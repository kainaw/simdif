import math
from ..simdif import Metric, METRICS, to_list


def info_bm25() -> str:
    return """
Okapi BM25 (Best Matching 25)
-----------------------------
A ranking function from probabilistic information retrieval that scores how well
a DOCUMENT (B) answers a QUERY (A), given the statistics of a CORPUS.

Inputs:
    A       - the query: a list of terms (a bare string becomes a char list).
    B       - the document being scored: a list of terms.
    corpus  - a list of documents (each a list of terms) used to derive the
              inverse document frequency (IDF) and the average document length.
              Passed as a keyword, like 'n_universe' for the set metrics. If
              omitted, the query and document (A and B) are used together as a
              small 2-document corpus, which makes IDF degenerate - always pass
              a real corpus for a meaningful score.

Formula (summed over the unique terms t of the query):
    score = sum_t  IDF(t) * ( tf(t,D) * (k1 + 1) )
                   / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        N     = number of documents in the corpus
        df(t) = number of corpus documents containing t
        tf    = term frequency of t in the document D
        |D|   = document length, avgdl = mean document length in the corpus
        k1    = term-frequency saturation (default 1.5)
        b     = length normalization in [0, 1] (default 0.75). Passed as
                'b_norm' here to avoid clashing with the document argument B.

Range: [0, inf)  - higher means more relevant. BM25 is asymmetric and corpus-
dependent, so it is a ranking score, not a distance. The '+1' inside the IDF
logarithm (the BM25+ variant) keeps IDF non-negative even for terms that occur
in more than half the corpus.

Note: order within A and B is ignored (bag-of-words); only term frequencies
matter. This is an educational single-document scoring of BM25 - a real search
engine ranks every document in a corpus against one query.

Aliases: Okapi, Okapi BM25
    """.strip()
info_okapi = info_bm25
info_okapi_bm25 = info_bm25


def _bm25_prepare(a, b, corpus):
    query = to_list(a)
    document = to_list(b)
    if not corpus:
        # No corpus supplied: fall back to the query and document themselves as
        # a small 2-document corpus so IDF/avgdl are at least defined.
        corpus = [query, document]
        degenerate = True
    else:
        corpus = [to_list(doc) for doc in corpus]
        degenerate = False
    n_docs = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n_docs if n_docs else 0.0
    return query, document, corpus, n_docs, avgdl, degenerate


def _bm25_idf(term, corpus, n_docs):
    df = sum(1 for doc in corpus if term in doc)
    return df, math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def explain_bm25(a, b, corpus=None, k1=1.5, b_norm=0.75, **kwargs) -> str:
    query, document, corpus, n_docs, avgdl, degenerate = _bm25_prepare(a, b, corpus)
    len_d = len(document)
    length_ratio = (len_d / avgdl) if avgdl > 0 else 0.0
    note = " (no corpus supplied -> using [query, document] as a 2-doc corpus; IDF is degenerate)" if degenerate else ""
    lines = []
    total = 0.0
    for term in dict.fromkeys(query):  # unique query terms, order preserved
        tf = document.count(term)
        df, idf = _bm25_idf(term, corpus, n_docs)
        if tf == 0:
            lines.append(f"  '{term}': tf=0 in document -> contributes 0 (idf={idf:.4f})")
            continue
        denom = tf + k1 * (1.0 - b_norm + b_norm * length_ratio)
        contrib = idf * (tf * (k1 + 1.0)) / denom
        total += contrib
        lines.append(
            f"  '{term}': df={df}, idf={idf:.4f}, tf={tf}, "
            f"denom={tf}+{k1}*(1-{b_norm}+{b_norm}*{len_d}/{avgdl:.4f})={denom:.4f} "
            f"-> {idf:.4f}*({tf}*{k1 + 1.0})/{denom:.4f} = {contrib:.4f}"
        )
    body = "\n".join(lines) if lines else "  (empty query)"
    return f"""
Query (A):    {query}
Document (B): {document}
Corpus: N={n_docs} documents, avgdl={avgdl:.4f}{note}
Parameters: k1={k1}, b={b_norm}
Per-query-term contributions:
{body}
BM25 score: {total:.4f}
    """.strip()
explain_okapi = explain_bm25
explain_okapi_bm25 = explain_bm25


@Metric
def score_bm25(a, b, corpus=None, k1=1.5, b_norm=0.75, **kwargs) -> float:
    query, document, corpus, n_docs, avgdl, _ = _bm25_prepare(a, b, corpus)
    length_ratio = (len(document) / avgdl) if avgdl > 0 else 0.0
    score = 0.0
    for term in set(query):
        tf = document.count(term)
        if tf == 0:
            continue
        _, idf = _bm25_idf(term, corpus, n_docs)
        denom = tf + k1 * (1.0 - b_norm + b_norm * length_ratio)
        score += idf * (tf * (k1 + 1.0)) / denom
    return score
score_okapi = score_bm25
score_okapi_bm25 = score_bm25


METRICS['bm25'] = {
    'class': 'sequence',
    'default': 'score',
    'score': score_bm25,
    'info': info_bm25,
    'explain': explain_bm25,
}
METRICS['okapi'] = METRICS['bm25']
METRICS['okapi_bm25'] = METRICS['bm25']
