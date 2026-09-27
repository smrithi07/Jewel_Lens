"""
Phase 8b — metric computation for the stumper eval.
"""

def recall_at_k(items, ground_truth_id, k):
    if not items:
        return False
    for item in items:
        top_ids = [r["catalogue_id"] for r in item["results"][:k]]
        if ground_truth_id in top_ids:
            return True
    return False


def mrr(items, ground_truth_id):
    if not items:
        return 0.0
    best = 0.0
    for item in items:
        for rank, r in enumerate(item["results"], 1):
            if r["catalogue_id"] == ground_truth_id:
                best = max(best, 1.0 / rank)
    return best


def category_from_catalogue_id(catalogue_id, metadata_lookup):
    return metadata_lookup.get(catalogue_id, {}).get("category", "")