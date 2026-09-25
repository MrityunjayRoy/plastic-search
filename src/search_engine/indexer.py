"""
This is a simple inverse index
"""

import time
import pickle
from typing import DefaultDict
from tokenizer import tokenize


class Indexer:
    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = DefaultDict(list)
        self.docs_lens: list[int] = []
        self.docs_ids: list[int] = []
        self.n_docs = 0
        self.total_len = 0

    def add(self, ext_id: int, text: str) -> None:
        tokens = tokenize(text)
        doc_id = self.n_docs
        tf: dict[str, int] = {}
        
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for t, f in tf.items():
            self.postings[t].append((doc_id, f))

        self.docs_lens.append(len(tokens))
        self.docs_ids.append(ext_id)
        self.n_docs += 1
        self.total_len += len(tokens)

def build(path: str, max_docs: int | None = None) -> tuple[Indexer, dict]:
    idx = Indexer()
    t0 = time.perf_counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\r").split("\t", 1)
            idx.add(int(pid), text)
            if max_docs is not None and idx.n_docs >= max_docs:
                break

        build_s = time.perf_counter() - t0
        return idx, {"build_s": build_s, "n_docs": idx.n_docs,
        "n_terms": len(idx.postings), "n_tokens": idx.total_len}

def save(idx: Indexer, path: str) -> float:
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        pickle.dump(
                {
                    "postings": dict(idx.postings), "docs_lens": idx.docs_lens,
                    "doc_ids": idx.docs_ids, "n_docs": idx.n_docs, "total_len": idx.total_len
                    },
                f, protocol=pickle.HIGHEST_PROTOCOL)
    return time.perf_counter() - t0

def load(path: str) -> tuple[Indexer, float]:
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        d = pickle.load(f)

    idx = Indexer()
    idx.postings = d["postings"]
    idx.docs_lens = d["doc_lens"]
    idx.docs_ids = d["doc_ids"]
    idx.n_docs = d["n_docs"]
    idx.total_len = d["total_len"]
    return idx, time.perf_counter() - t0
