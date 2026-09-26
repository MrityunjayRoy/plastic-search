import argparse
import os

from . import engine, indexer

DEFAULT_MAX_DOCS = 100_000


def _build(args: argparse.Namespace) -> None:
    idx, stats = indexer.build(args.collection, max_docs=args.max_docs)
    save_s = indexer.save(idx, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"indexed {stats['n_docs']} docs, {stats['n_terms']} terms, "
          f"{stats['n_tokens']} tokens")
    print(f"build {stats['build_s']:.1f}s  save {save_s:.1f}s  "
          f"wrote {args.output} ({size_mb:.1f} MB)")


def _search(args: argparse.Namespace) -> None:
    idx, load_s = indexer.load(args.index)
    results = engine.search(idx, args.query, k=args.k)
    print(f"{idx.n_docs} docs loaded in {load_s:.1f}s, "
          f"{len(results)} hit(s) for {args.query!r}")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:>3}. doc {doc_id:<10} {score:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="search-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build an index from a TSV collection")
    b.add_argument("collection", help="path to collection.tsv")
    b.add_argument("-o", "--output", default="index.pkl", help="output pickle path")
    b.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                   help=f"stop after N docs (default {DEFAULT_MAX_DOCS})")
    b.set_defaults(func=_build)

    s = sub.add_parser("search", help="query a built index")
    s.add_argument("index", help="path to index pickle")
    s.add_argument("query", help="query string")
    s.add_argument("-k", type=int, default=10, help="number of results (default 10)")
    s.set_defaults(func=_search)

    args = parser.parse_args(argv)
    args.func(args)
