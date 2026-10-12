"""
Phase 7 — consolidated single-image pipeline.
Wraps detect -> crop -> embed -> classify -> retrieve -> rerank -> decide
into one function, so multi-item handling is just "call this once per
detected box" rather than special-cased logic. Reused by Phase 8's eval
harness and any future API/frontend layer.
"""

import sys
sys.path.append(".")
import config
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.embed import embed_image_with_raw, load_embedder
from src.pipeline.retrieve import search, apply_rerank, load_index
from src.pipeline.classify import classify, load_classifier
from src.pipeline.decision import decide
from src.pipeline.category import categorize, load_category_models
from src.pipeline.recommend import mmr, split_tiers, out_of_scope_prob


def load_all_models():
    """Call once at startup — loads detector, embedder, index, classifier."""
    load_detector()
    load_embedder()
    load_index()
    load_classifier()
    if config.CATEGORY_MODE == "fused":
        load_category_models()


def run_pipeline(image, top_k=5):
    """
    image: PIL Image (RGB)
    Returns: list of dicts, one per detected item:
        {
            item_index, box, crop,
            detector_category, detector_score,
            classifier_category, classifier_confidence,
            results (top-k after rerank),
            decision: {status, top_result, similarity, margin}
        }
    Empty list means no jewelry detected at all.
    """
    detections = detect(image)
    items = []

    for i, d in enumerate(detections):
        crop = crop_with_padding(image, d["box"])
        query_embedding, raw_embedding = embed_image_with_raw(crop)

        results = search(query_embedding, top_k=top_k)
        p_oos = None
        if config.CATEGORY_MODE == "fused":
            fused = categorize(crop, query_embedding, raw_embedding, d["category"], results)
            category_probs, predicted_category, confidence = fused["probs"], fused["category"], fused["confidence"]
            if fused.get("clip_emb") is not None:
                p_oos = out_of_scope_prob(fused["clip_emb"])
        else:
            category_probs, predicted_category, confidence = classify(query_embedding)
        results = apply_rerank(results, category_probs)
        decision = decide(results, p_oos=p_oos)            # decided on the relevance order, before diversity
        results = mmr(results)
        closest, similar = split_tiers(results)

        items.append({
            "item_index": i,
            "box": d["box"],
            "crop": crop,
            "detector_category": d["category"],
            "detector_score": d["score"],
            "classifier_category": predicted_category,
            "classifier_confidence": confidence,
            "results": results,
            "closest_matches": closest,
            "similar_items": similar,
            "decision": decision,
        })

    return items