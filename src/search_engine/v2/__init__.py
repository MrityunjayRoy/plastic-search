import argparse
import json
import os
import time

from . import indexer
from .engine import SearcherV2

DEFAULT_INDEX_DIR = "index"


def _build(args: argparse.Namespace) -> None:
    stats = indexer.build(args.collection, args.out_dir,
                          max_docs=args.max_docs)
    size_mb = sum(e.stat().st_size for e in os.scandir(args.out_dir)) / 1e6

    print(f"indexed {stats['n_docs']:,} docs, {stats['n_terms']:,} terms, "
          f"{stats['total_postings']:,} postings (avgdl {stats['avgdl']:.1f})")
    print(f"scan {stats['scan_s']:.1f}s  assemble {stats['assemble_s']:.1f}s  "
          f"impacts {stats['impacts_s']:.1f}s  write {stats['write_s']:.1f}s")
    print(f"wrote {args.out_dir} ({size_mb:.1f} MB)")
    print(json.dumps(stats, indent=2))


def _search(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    searcher = SearcherV2(args.index)
    load_s = time.perf_counter() - t0

    results = searcher.search(args.query, k=args.k)
    print(f"{searcher.n_docs:,} docs, {searcher.meta['n_terms']:,} terms, "
          f"{searcher.meta['total_postings']:,} postings "
          f"loaded in {load_s:.1f}s")
    print(f"{len(results)} hit(s) for {args.query!r}")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:>3}. doc {doc_id:<10} {score:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="search-engine-v2")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build an index from a TSV collection")
    b.add_argument("collection", help="path to collection.tsv")
    b.add_argument("out_dir", nargs="?", default=DEFAULT_INDEX_DIR,
                   help=f"output index directory (default {DEFAULT_INDEX_DIR})")
    b.add_argument("--max-docs", type=int, default=None,
                   help="stop after N docs (default: whole collection)")
    b.set_defaults(func=_build)

    s = sub.add_parser("search", help="query a built index")
    s.add_argument("index", help="path to index directory")
    s.add_argument("query", help="query string")
    s.add_argument("-k", type=int, default=10,
                   help="number of results (default 10)")
    s.set_defaults(func=_search)

    args = parser.parse_args(argv)
    args.func(args)
