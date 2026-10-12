"""
Phase 6 — confidence scoring and no-match decision.
Combines top-1 similarity and the top1-top2 margin (not classifier
confidence — kept simple and defensible, per the trimmed architecture).
"""

import sys
sys.path.append(".")
import config


def decide(results, p_oos=None):
    """p_oos: optional CLIP out-of-scope probability of the crop (Phase 8); above
    config.OOS_THRESHOLD the item is treated as not one of our categories."""
    if not results:
        return {"status": "no_match", "top_result": None, "similarity": 0.0, "margin": 0.0, "reason": "no_results"}

    top1 = results[0]
    top2 = results[1] if len(results) > 1 else None
    similarity = float(top1["score"])
    margin = float(similarity - top2["score"]) if top2 else similarity

    if p_oos is not None and p_oos > getattr(config, "OOS_THRESHOLD", 1.01):
        status, reason = "no_match", "out_of_scope"
    elif similarity < config.LOW_MATCH_THRESHOLD:
        status, reason = "no_match", "low_similarity"
    elif similarity < config.HIGH_MATCH_THRESHOLD or margin < config.MARGIN_THRESHOLD:
        status, reason = "uncertain", "below_closest_threshold"
    else:
        status, reason = "match", "closest_match"

    return {
        "status": status,
        "reason": reason,
        "p_oos": None if p_oos is None else round(float(p_oos), 4),
        "top_result": top1,
        "similarity": round(similarity, 4),
        "margin": round(margin, 4),
    }