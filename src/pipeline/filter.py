"""
Phase 9 — price/relevance filter and sort, applied to retrieval results
after the decision step. Pure post-processing on results already
returned by run_pipeline() — no re-embedding, no re-search.
"""

import pandas as pd
import sys
sys.path.append(".")
import config

_metadata_cache = None


def load_metadata():
    """Cached lookup: catalogue_id -> {price, title, listing_url}."""
    global _metadata_cache
    if _metadata_cache is None:
        df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
        _metadata_cache = df.set_index("catalogue_id")[["price", "title", "listing_url"]].to_dict("index")
    return _metadata_cache


def enrich_with_metadata(results):
    """Attach price/title/listing_url to each result dict."""
    meta = load_metadata()
    enriched = []
    for r in results:
        info = meta.get(r["catalogue_id"], {})
        enriched.append({
            **r,
            "price": info.get("price"),
            "title": info.get("title", ""),
            "listing_url": info.get("listing_url", ""),
        })
    return enriched


def filter_and_sort(results, min_price=None, max_price=None, sort_by="relevance"):
    """
    results: list of result dicts (already enriched or not — enriches if needed)
    sort_by: "relevance" (default, by similarity score) or "price_low" or "price_high"
    """
    results = enrich_with_metadata(results)

    if min_price is not None:
        results = [r for r in results if r["price"] is not None and r["price"] >= min_price]
    if max_price is not None:
        results = [r for r in results if r["price"] is not None and r["price"] <= max_price]

    if sort_by == "price_low":
        results = sorted(results, key=lambda r: r["price"] if r["price"] is not None else float("inf"))
    elif sort_by == "price_high":
        results = sorted(results, key=lambda r: r["price"] if r["price"] is not None else float("-inf"), reverse=True)
    # "relevance" — already sorted by similarity from retrieve.py, leave as-is

    return results