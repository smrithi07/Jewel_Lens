"""
Phase 7 — consolidated single-image pipeline.
Wraps detect -> crop -> embed -> classify -> retrieve -> rerank -> decide
into one function, so multi-item handling is just "call this once per
detected box" rather than special-cased logic. Reused by Phase 8's eval
harness and any future API/frontend layer.
"""

import sys
sys.path.append(".")
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.embed import embed_image, load_embedder
from src.pipeline.retrieve import search, apply_rerank, load_index
from src.pipeline.classify import classify, load_classifier
from src.pipeline.decision import decide


def load_all_models():
    """Call once at startup — loads detector, embedder, index, classifier."""
    load_detector()
    load_embedder()
    load_index()
    load_classifier()


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
        query_embedding = embed_image(crop)

        category_probs, predicted_category, confidence = classify(query_embedding)
        results = search(query_embedding, top_k=top_k)
        results = apply_rerank(results, category_probs)
        decision = decide(results)

        items.append({
            "item_index": i,
            "box": d["box"],
            "crop": crop,
            "detector_category": d["category"],
            "detector_score": d["score"],
            "classifier_category": predicted_category,
            "classifier_confidence": confidence,
            "results": results,
            "decision": decision,
        })

    return items