"""
Phase 6 — confidence scoring and no-match decision.
Combines top-1 similarity and the top1-top2 margin (not classifier
confidence — kept simple and defensible, per the trimmed architecture).
"""

import sys
sys.path.append(".")
import config


def decide(results):
    if not results:
        return {"status": "no_match", "top_result": None, "similarity": 0.0, "margin": 0.0}

    top1 = results[0]
    top2 = results[1] if len(results) > 1 else None
    similarity = float(top1["score"])
    margin = float(similarity - top2["score"]) if top2 else similarity

    if similarity < config.LOW_MATCH_THRESHOLD:
        status = "no_match"
    elif similarity < config.HIGH_MATCH_THRESHOLD or margin < config.MARGIN_THRESHOLD:
        status = "uncertain"
    else:
        status = "match"

    return {
        "status": status,
        "top_result": top1,
        "similarity": round(similarity, 4),
        "margin": round(margin, 4),
    }