"""Embed `corpus/` into the local ChromaDB index used by the `rag_search` tool.

Usage:
    python -m evals.build_index            # writes to $CHROMA_DIR (default .chroma)
    python -m evals.build_index --check    # also runs a few retrieval sanity queries
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from tools.mcp_server.rag import DEFAULT_CORPUS_DIR, Retriever, build_index, chroma_dir

# (query, doc that must be the top hit) - a quick smoke test of retrieval quality.
SANITY_QUERIES = [
    ("capital city of Canada", "capital-canada"),
    ("why is Mars red", "planet-mars"),
    ("first US patent for the telephone", "invention-telephone"),
    ("deepest point in the ocean", "geo-pacific"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run retrieval sanity queries after building")
    args = parser.parse_args()

    load_dotenv()
    target = chroma_dir()
    n = build_index(DEFAULT_CORPUS_DIR, target)
    print(f"Indexed {n} chunks from {DEFAULT_CORPUS_DIR.name}/ into {target}")

    if args.check:
        retriever = Retriever(target)
        failures = 0
        for query, expected in SANITY_QUERIES:
            top = retriever.search(query, top_k=1)[0]
            ok = top["doc_id"] == expected
            failures += not ok
            print(f"  [{'ok' if ok else 'FAIL'}] {query!r} -> {top['doc_id']} (score {top['score']})")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
