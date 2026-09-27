"""
Phase 6 visual check — runs the same pipeline as scratch_phase6_test.py
but generates an HTML report showing the query crop next to its top-5
catalogue matches (as images, not paths), for both positive and
negative test sets. Open the output HTML in a browser to eyeball results.
"""

import os
import base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
from io import BytesIO
import sys
sys.path.append(".")
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.embed import embed_image, load_embedder
from src.pipeline.retrieve import search, apply_rerank, load_index
from src.pipeline.classify import classify, load_classifier
from src.pipeline.decision import decide

POSITIVE_DIR = "data/detect_test"
NEGATIVE_DIR = "data/negatives"
OUTPUT_HTML = "data/phase6_visual_report.html"

STATUS_COLORS = {"match": "#2e7d32", "uncertain": "#f9a825", "no_match": "#c62828"}


def image_to_base64(image, max_size=200):
    image = image.copy()
    image.thumbnail((max_size, max_size))
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(image, label=""):
    b64 = image_to_base64(image)
    return f'<div class="thumb"><img src="data:image/jpeg;base64,{b64}"><div class="caption">{label}</div></div>'


def run_on_dir(dir_path, label, html_parts):
    if not os.path.isdir(dir_path):
        return
    images = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    html_parts.append(f'<h2>{label}</h2>')

    for fname in images:
        path = os.path.join(dir_path, fname)
        image = Image.open(path).convert("RGB")
        detections = detect(image)

        html_parts.append(f'<h3>{fname} — {len(detections)} detection(s)</h3>')
        if not detections:
            html_parts.append('<p style="color:#c62828">NO_MATCH — no jewelry detected</p>')
            continue

        for i, d in enumerate(detections):
            crop = crop_with_padding(image, d["box"])
            query_embedding = embed_image(crop)
            category_probs, pred_cat, conf = classify(query_embedding)
            results = search(query_embedding, top_k=5)
            results = apply_rerank(results, category_probs)
            decision = decide(results)

            color = STATUS_COLORS[decision["status"]]
            html_parts.append(
                f'<div class="row">'
                f'{img_tag(crop, f"QUERY (Item {i+1})")}'
                f'<div class="status" style="color:{color}">'
                f'{decision["status"].upper()}<br>sim={decision["similarity"]:.3f} margin={decision["margin"]:.3f}'
                f'<br>detector: {d["category"]} | classifier: {pred_cat} ({conf:.2f})</div>'
            )
            for rank, r in enumerate(results, 1):
                try:
                    result_img = Image.open(r["image_path"]).convert("RGB")
                    html_parts.append(img_tag(result_img, f"#{rank} sim={r['score']:.3f}\n{r['category']}"))
                except Exception as e:
                    html_parts.append(f'<div class="thumb">[missing: {e}]</div>')
            html_parts.append('</div>')


def main():
    load_detector()
    load_embedder()
    load_index()
    load_classifier()

    html_parts = ["""
    <html><head><style>
    body { font-family: sans-serif; margin: 20px; }
    .row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 20px;
           border-bottom: 1px solid #ddd; padding-bottom: 15px; flex-wrap: wrap; }
    .thumb { text-align: center; }
    .thumb img { width: 150px; height: 150px; object-fit: cover; border: 1px solid #ccc; }
    .caption { font-size: 11px; white-space: pre-line; max-width: 150px; }
    .status { font-weight: bold; min-width: 150px; }
    </style></head><body>
    """]

    run_on_dir(POSITIVE_DIR, "POSITIVE TEST SET", html_parts)
    run_on_dir(NEGATIVE_DIR, "NEGATIVE TEST SET", html_parts)

    html_parts.append("</body></html>")

    with open(OUTPUT_HTML, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Report saved to {OUTPUT_HTML} — open it in a browser.")


if __name__ == "__main__":
    main()