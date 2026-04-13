"""CLI demo for hybrid entity search.

Usage
-----
    # Run the pipeline first if gold/entity_lakehouse.duckdb does not exist:
    python scripts/run_demo.py

    # Then run hybrid search queries:
    python scripts/search_demo.py "solar energy Germany"
    python scripts/search_demo.py "infrastructure holding" --top-k 3
    python scripts/search_demo.py "wind operator"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path so the package is importable without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from entity_data_lakehouse.search import build_search_index  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + dense vector entity search demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="Free-text search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--duckdb",
        type=Path,
        default=_REPO_ROOT / "gold" / "entity_lakehouse.duckdb",
        metavar="PATH",
        help="Path to entity_lakehouse.duckdb (default: gold/entity_lakehouse.duckdb).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.duckdb.exists():
        print(
            f"DuckDB file not found: {args.duckdb}\n"
            "Run the pipeline first:\n"
            "    python scripts/run_demo.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building hybrid search index from {args.duckdb} ...")
    index = build_search_index(args.duckdb)

    print(f'\nQuery: "{args.query}"  top_k={args.top_k}\n')
    print(f"{'Rank':<5} {'RRF Score':<12} {'BM25↑':<7} {'Vec↑':<7} {'Entity':<35} {'Country':<8} {'Type'}")
    print("-" * 100)

    results = index.search(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    for rank, r in enumerate(results, start=1):
        bm25_display = str(r.bm25_rank) if r.bm25_rank is not None else "—"
        vec_display = str(r.vector_rank) if r.vector_rank is not None else "—"
        print(
            f"{rank:<5} {r.rrf_score:<12.6f} {bm25_display:<7} {vec_display:<7} "
            f"{r.entity_name[:34]:<35} {r.country_code:<8} {r.entity_type}"
        )


if __name__ == "__main__":
    main()
