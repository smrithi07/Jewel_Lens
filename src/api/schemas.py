"""
Response shape for /predict — documented here so the API contract is
explicit rather than implied by whatever run_pipeline() happens to return.
"""

import sys
sys.path.append(".")
import config


def extract_shop_name(catalogue_id):
    """catalogue_id format: shopify_<domain>_<product_id> -> readable shop name."""
    parts = catalogue_id.split("_")
    if len(parts) >= 2 and parts[0] == "shopify":
        domain = parts[1]
        return domain.replace(".com", "").replace(".in", "").replace(".co", "").title()
    return "Unknown"


def build_result_item(result, price, title, listing_url):
    return {
        "catalogue_id": result["catalogue_id"],
        "category": result["category"],
        "similarity": round(result["score"], 4),
        "shop": extract_shop_name(result["catalogue_id"]),
        "price": price,
        "title": title,
        "listing_url": listing_url,
        "image_url": f"/catalogue-image/{result['image_path']}",
    }


def crop_to_data_url(crop, max_side=220):
    """Small JPEG of the detected crop, so the frontend shows exactly what was detected."""
    import base64
    from io import BytesIO
    if crop is None:
        return None
    img = crop.convert("RGB").copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_item_response(pipeline_item, enriched_results, closest=None, similar=None):
    """enriched_results: top results after the price filter / sort; closest / similar: the same, split into
    the Phase 8 tiers ("Closest matches" score >= HIGH_MATCH_THRESHOLD, "Similar items" the rest)."""
    d = pipeline_item["decision"]
    return {
        "item_index": pipeline_item["item_index"],
        # fused mode: show the fused category (Phase 4); legacy: the detector's label, as before
        "detected_category": (pipeline_item["classifier_category"] if config.CATEGORY_MODE == "fused"
                              else pipeline_item["detector_category"]),
        "category_confidence": round(float(pipeline_item["classifier_confidence"]), 4),
        "detector_category": pipeline_item["detector_category"],
        "detector_confidence": round(pipeline_item["detector_score"], 4),
        "crop_image": crop_to_data_url(pipeline_item.get("crop")),
        "status": d["status"],
        "status_reason": d.get("reason"),
        "out_of_scope_probability": d.get("p_oos"),
        "closest_matches": closest if closest is not None else [],
        "similar_items": similar if similar is not None else [],
        "similarity": d["similarity"],
        "margin": d["margin"],
        "results": enriched_results,
    }