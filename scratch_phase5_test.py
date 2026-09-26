"""
Throwaway Phase 5 checkpoint script — runs the full pipeline WITH
classifier re-rank, on the same test photos from Phase 4, so you can
compare before/after directly.
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

TEST_DIR = "data/detect_test"


def main():
    load_detector()
    load_embedder()
    load_index()
    load_classifier()

    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in test_images:
        path = os.path.join(TEST_DIR, fname)
        image = Image.open(path).convert("RGB")
        detections = detect(image)

        print(f"\n{'='*60}")
        print(f"{fname}: {len(detections)} detection(s)")
        if not detections:
            print("  No jewelry detected.")
            continue

        for i, d in enumerate(detections):
            crop = crop_with_padding(image, d["box"])
            query_embedding = embed_image(crop)

            category_probs, predicted_category, confidence = classify(query_embedding)
            results_before = search(query_embedding, top_k=5)
            results_after = apply_rerank(results_before, category_probs)

            print(f"\n  Item {i+1}: detector said '{d['category']}' | "
                  f"classifier says '{predicted_category}' (conf {confidence:.3f})")
            print(f"    Top-5 BEFORE re-rank: {[r['category'] for r in results_before]}")
            print(f"    Top-5 AFTER  re-rank: {[r['category'] for r in results_after]}")
            for rank, r in enumerate(results_after, 1):
                print(f"      #{rank} sim={r['score']:.3f} | {r['category']} | {r['catalogue_id']}")


if __name__ == "__main__":
    main()