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
        image = Image.open(file.stream)
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
        top5 = filtered[:5]
        enriched = [
            build_result_item(r, r["price"], r["title"], r["listing_url"])
            for r in top5
        ]
        response_items.append(build_item_response(item, enriched))

    return jsonify({"items": response_items})

@app.route("/catalogue-image/<path:image_path>")
def catalogue_image(image_path):
    from flask import send_from_directory
    full_path = os.path.join(os.getcwd(), image_path)
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)