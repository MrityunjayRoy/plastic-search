"""
a simple BM-25 query engine, term-at-a-time, top-k results

k1 and b are fixed according to ANSERINI
"""

k1 = 0.9
b = 0.4

import math
import heapq
from .indexer import Indexer
from .tokenizer import tokenize

def search(idx: Indexer, query: str, k: int = 10) -> list[tuple[int, float]]:
    terms = tokenize(query)
    if not terms or idx.n_docs == 0:
        return []

    avgdl = idx.total_len/idx.n_docs
    top_scores: dict[int, float] = {}

    for t in set(terms):
        plists = idx.postings.get(t)
        if not plists:
            continue
        df = len(plists)
        idf = math.log(1.0 + (idx.n_docs - df + 0.5) / (df + 0.5))

        for docid, tf in plists:
            dl = idx.docs_lens[docid]
            s = idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl/avgdl))
            if docid in top_scores:
                top_scores[docid] += s 
            else:
                top_scores[docid] = s

    top_docs = heapq.nlargest(k, top_scores.items(), key=lambda kv : kv[1])
    
    return [(idx.docs_ids[d], s) for d, s in top_docs]
