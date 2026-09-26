"""
Response shape for /predict — documented here so the API contract is
explicit rather than implied by whatever run_pipeline() happens to return.
"""

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


def build_item_response(pipeline_item, enriched_results):
    d = pipeline_item["decision"]
    return {
        "item_index": pipeline_item["item_index"],
        "detected_category": pipeline_item["detector_category"],
        "detector_confidence": round(pipeline_item["detector_score"], 4),
        "status": d["status"],
        "similarity": d["similarity"],
        "margin": d["margin"],
        "results": enriched_results,
    }