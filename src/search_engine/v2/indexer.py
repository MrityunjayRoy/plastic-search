import os
import json
import pickle
import numpy as np
import time
from array import array

from .tokenizer import tokenize

k1 = 0.9
b = 0.4
TF_CLAMP = 63

def build(path: str, out_dir: str, max_docs: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    timings: dict = {}

    term_buffs: dict[str, array] = {} 
    doc_lens = array("I")
    t0 = time.perf_counter()
    n_docs = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            assert int(pid) == n_docs, f"pids not sequential at line {n_docs}"

            toks = tokenize(text)
            tf: dict[str, int] = {}
            get = tf.get
            for t in toks:
                tf[t] = get(t, 0) + 1

            packed_doc = n_docs << 6

            for t, c in tf.items():
                buf = term_buffs.get(t)
                if buf is None:
                    buf = term_buffs[t] = array("I")

                buf.append(packed_doc | min(c, TF_CLAMP))
            doc_lens.append(len(toks))
            n_docs += 1
            if max_docs is not None and n_docs >= max_docs:
                break

        timings["scan_s"] = time.perf_counter() - t0

    # assembling
    t0 = time.perf_counter()
    n_terms = len(term_buffs)
    dfs = np.empty(n_terms, np.int64)
    terms = {}
    for tid, (term, buf) in enumerate(term_buffs.items()):
        terms[term] = tid
        dfs[tid] = len(buf)
    offsets = np.zeros(n_terms + 1, np.uint64)
    np.cumsum(dfs, out=offsets[1:])
    total = int(offsets[-1])

    doc_ids = np.empty(total, np.uint32)
    tfs = np.empty(total, np.uint8)
    pos = 0
    for term, buf in term_buffs.items():
        v = np.frombuffer(buf, dtype=np.uint32)
        n = len(v)
        doc_ids[pos:pos + n] = v >> 6 
        tfs[pos:pos + n] = (v & TF_CLAMP).astype(np.uint8)
        pos += n
    term_buffs.clear()
    timings["assemble_s"] = time.perf_counter() - t0

    # impacts
    t0 = time.perf_counter()
    dl = np.asarray(doc_lens, dtype=np.float32)
    avgdl = float(np.mean(dl))
    idf_t = np.log(1.0 + (n_docs - dfs + 0.5) / (dfs + 0.5)).astype(np.float32)
    impacts = np.empty(total, np.float32)
    idf_p = np.repeat(idf_t, dfs)
    CH = 50_000_000

    for s in range(0, total, CH):
        e = min(s + CH, total)
        tf_f = tfs[s:e].astype(np.float32)
        dl_p = dl[doc_ids[s:e]]
        impacts[s:e] = idf_p[s:e] * tf_f * (k1 + 1.0) / (
                tf_f + k1 * (1.0 - b + b * dl_p / avgdl ))
                
    del idf_p
    timings["impacts_s"] = time.perf_counter() - t0

    # save to disk
    t0 = time.perf_counter()
    np.save(f"{out_dir}/offsets.u64.npy", offsets)
    np.save(f"{out_dir}/docids.u32.npy", doc_ids)
    np.save(f"{out_dir}/tfs.u8.npy", tfs)
    np.save(f"{out_dir}/impacts.f32.npy", impacts)
    np.save(f"{out_dir}/doclens.u32.npy", np.asarray(doc_lens, np.uint32))
    with open(f"{out_dir}/terms.pkl", "wb") as f:
        pickle.dump(terms, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n_docs": n_docs, "n_terms": n_terms, "total_postings": total,
            "avgdl": avgdl, "k1": k1, "b": b}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    timings["write_s"] = time.perf_counter() - t0
    timings.update(meta)
    return timings

if __name__ == "__main__":
    import sys
    stats = build(sys.argv[1], sys.argv[2],
                  int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(json.dumps(stats, indent=2))
