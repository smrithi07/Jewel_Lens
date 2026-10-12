"""
Phase 9.5 — Flask API wrapping the full pipeline.
POST /predict with an image file -> JSON with one entry per detected
item, each carrying its top-N matches (price, shop, title included).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
from PIL import Image
import sys
sys.path.append(".")
from src.pipeline.run_pipeline import run_pipeline, load_all_models
from src.pipeline.filter import filter_and_sort, load_metadata
from src.api.schemas import build_result_item, build_item_response
from src.pipeline.recommend import similar_to
from src.pipeline.preprocess import preprocess_image
import config

app = Flask(__name__, static_folder="../frontend/static", static_url_path="")

print("Loading models (this happens once at startup)...")
load_all_models()
print("Models loaded — API ready.")


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        # EXIF rotation + 1600 px cap (v1 skipped this, so rotated phone photos were processed sideways)
        image = preprocess_image(Image.open(file.stream))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    sort_by = request.args.get("sort_by", default="relevance")

    pipeline_items = run_pipeline(image, top_k=20)

    if not pipeline_items:
        return jsonify({"items": [], "message": "No jewelry detected in this photo."})

    response_items = []
    for item in pipeline_items:
        filtered = filter_and_sort(item["results"], min_price=min_price, max_price=max_price, sort_by=sort_by)
        enriched_all = [build_result_item(r, r["price"], r["title"], r["listing_url"]) for r in filtered]
        closest, similar = tiers(enriched_all)
        response_items.append(build_item_response(item, enriched_all[:5], closest[:5], similar[:5]))

    return jsonify({"items": response_items})


def tiers(enriched):
    """Phase 8 tiers on API result dicts (their 'similarity' is the post-re-rank score)."""
    t = config.HIGH_MATCH_THRESHOLD
    return [r for r in enriched if r["similarity"] >= t], [r for r in enriched if r["similarity"] < t]


@app.route("/similar/<catalogue_id>")
def similar(catalogue_id):
    """'More like this': recommendations for a catalogue listing from its stored embeddings.
    Same price filter / sort query parameters as /predict."""
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    sort_by = request.args.get("sort_by", default="relevance")
    results, category = similar_to(catalogue_id)
    if results is None:
        return jsonify({"error": f"Unknown catalogue_id: {catalogue_id}"}), 404
    filtered = filter_and_sort(results, min_price=min_price, max_price=max_price, sort_by=sort_by)
    enriched = [build_result_item(r, r["price"], r["title"], r["listing_url"]) for r in filtered]
    closest, similar_items = tiers(enriched)
    info = load_metadata().get(catalogue_id, {})
    return jsonify({"query": {"catalogue_id": catalogue_id, "category": category, "title": info.get("title"),
                              "price": None if info.get("price") is None else float(info["price"])},
                    "results": enriched[:5], "closest_matches": closest[:5], "similar_items": similar_items[:5]})

@app.route("/catalogue-image/<path:image_path>")
def catalogue_image(image_path):
    from flask import send_from_directory, abort
    # only catalogue images: v1 served any file under the project (e.g. .env, config.py)
    prefix = config.RAW_IMAGE_DIR.rstrip("/") + "/"
    if not image_path.startswith(prefix):
        abort(404)
    return send_from_directory(os.path.join(os.getcwd(), config.RAW_IMAGE_DIR), image_path[len(prefix):])


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))   # PORT=5001 when AirPlay holds 5000