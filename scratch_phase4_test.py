import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import sys
sys.path.append(".")
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.embed import embed_image, load_embedder
from src.pipeline.retrieve import search, load_index

TEST_DIR = "data/detect_test"


def main():
    load_detector()
    load_embedder()
    load_index()

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
            results = search(query_embedding, top_k=5)

            print(f"\n  Item {i+1}: detected as '{d['category']}' (detector score {d['score']:.3f})")
            if not results:
                print("    No matches found.")
                continue
            for rank, r in enumerate(results, 1):
                print(f"    #{rank} sim={r['score']:.3f} | {r['category']} | {r['catalogue_id']} | {r['image_path']}")


if __name__ == "__main__":
    main()