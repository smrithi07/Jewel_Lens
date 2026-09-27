"""
Throwaway Phase 6 checkpoint script — runs the full pipeline including
the no-match decision, on both your known-positive test photos
(data/detect_test/) and negative photos (data/negatives/ — items NOT
in your catalogue: a different accessory, a non-jewelry object, etc.)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import sys
sys.path.append(".")
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.embed import embed_image, load_embedder
from src.pipeline.retrieve import search, apply_rerank, load_index
from src.pipeline.classify import classify, load_classifier
from src.pipeline.decision import decide

POSITIVE_DIR = "data/detect_test"
NEGATIVE_DIR = "data/negatives"


def run_on_dir(dir_path, label):
    if not os.path.isdir(dir_path):
        print(f"\n[skip] {dir_path} doesn't exist yet")
        return

    images = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\n{'#'*60}\n{label} ({len(images)} images)\n{'#'*60}")

    for fname in images:
        path = os.path.join(dir_path, fname)
        image = Image.open(path).convert("RGB")
        detections = detect(image)

        print(f"\n{fname}: {len(detections)} detection(s)")
        if not detections:
            print("  -> NO_MATCH (no jewelry detected at all)")
            continue

        for i, d in enumerate(detections):
            crop = crop_with_padding(image, d["box"])
            query_embedding = embed_image(crop)
            category_probs, _, _ = classify(query_embedding)
            results = search(query_embedding, top_k=5)
            results = apply_rerank(results, category_probs)
            decision = decide(results)

            print(f"  Item {i+1}: sim={decision['similarity']:.3f} margin={decision['margin']:.3f} "
                  f"-> {decision['status'].upper()}"
                  + (f" ({decision['top_result']['catalogue_id']})" if decision['top_result'] else ""))


def main():
    load_detector()
    load_embedder()
    load_index()
    load_classifier()

    run_on_dir(POSITIVE_DIR, "POSITIVE TEST SET (items believed in-catalogue)")
    run_on_dir(NEGATIVE_DIR, "NEGATIVE TEST SET (items NOT in catalogue)")


if __name__ == "__main__":
    main()