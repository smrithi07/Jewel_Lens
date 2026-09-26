"""
Phase 4b — FAISS retrieval over the catalogue index.
Searches top-N raw hits, then dedupes by catalogue_id (since a listing
can have multiple images indexed) to return top-K distinct items.
"""

import faiss
import pickle
import numpy as np
import sys
sys.path.append(".")
import config

_index = None
_id_mapping = None


def load_index():
    global _index, _id_mapping
    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.ID_MAPPING_PATH, "rb") as f:
            _id_mapping = pickle.load(f)
    return _index, _id_mapping


def search(query_embedding, top_k=20, raw_k=40):
    """
    query_embedding: normalized 1D numpy vector (float32)
    Returns: list of dicts {catalogue_id, category, image_path, score},
    deduped by catalogue_id, best score kept per item, top_k results.
    """
    index, id_mapping = load_index()
    query = query_embedding.reshape(1, -1)

    scores, result_ids = index.search(query, raw_k)

    seen_catalogue_ids = {}
    for score, faiss_id in zip(scores[0], result_ids[0]):
        if faiss_id == -1:
            continue
        meta = id_mapping[int(faiss_id)]
        cat_id = meta["catalogue_id"]
        if cat_id not in seen_catalogue_ids or score > seen_catalogue_ids[cat_id]["score"]:
            seen_catalogue_ids[cat_id] = {
                "catalogue_id": cat_id,
                "category": meta["category"],
                "image_path": meta["image_path"],
                "score": float(score),
            }

    ranked = sorted(seen_catalogue_ids.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

def apply_rerank(results, category_probs):
    """
    Soft re-rank: boost each candidate's score by RERANK_BOOST_WEIGHT
    scaled by the classifier's confidence for that candidate's category.
    Only applies if the top predicted category's confidence clears
    RERANK_MIN_CONFIDENCE — otherwise the classifier's vote is too
    uncertain to trust, so results pass through unchanged.
    """
    top_category = max(category_probs, key=category_probs.get)
    top_confidence = category_probs[top_category]

    if top_confidence < config.RERANK_MIN_CONFIDENCE:
        return results  # classifier unsure — don't apply any boost

    reranked = []
    for r in results:
        boost = config.RERANK_BOOST_WEIGHT * category_probs.get(r["category"], 0.0)
        reranked.append({**r, "score": r["score"] + boost, "rerank_boost": boost})

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked