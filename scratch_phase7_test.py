"""
Throwaway Phase 7 checkpoint script — runs run_pipeline() on multi-item
photos specifically, and checks that detected items are genuinely
independent: distinct boxes (no duplicate/overlapping crops slipping
through), and each producing its own top-5 rather than an identical
copy of another item's results.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import sys
sys.path.append(".")
from src.pipeline.run_pipeline import run_pipeline, load_all_models

TEST_DIR = "data/detect_test"


def boxes_overlap_significantly(box1, box2, iou_threshold=0.5):
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    if x1 <= x0 or y1 <= y0:
        return False
    inter = (x1 - x0) * (y1 - y0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter)
    return iou > iou_threshold


def main():
    load_all_models()

    all_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    multi_item_results = {}

    # first pass: find which images actually have 2+ detections
    for fname in all_images:
        path = os.path.join(TEST_DIR, fname)
        image = Image.open(path).convert("RGB")
        items = run_pipeline(image)
        if len(items) >= 2:
            multi_item_results[fname] = items

    print(f"Found {len(multi_item_results)} multi-item photos out of {len(all_images)} total\n")

    total_checks = 0
    total_passed = 0

    for fname, items in multi_item_results.items():
        print(f"{'='*60}")
        print(f"{fname}: {len(items)} items detected")

        # check 1: no two boxes overlap significantly (real distinct items, not dup detections)
        boxes = [item["box"] for item in items]
        overlap_found = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes_overlap_significantly(boxes[i], boxes[j]):
                    print(f"  [WARN] Item {i} and Item {j} boxes overlap significantly — possible duplicate detection")
                    overlap_found = True
        total_checks += 1
        if not overlap_found:
            total_passed += 1
            print(f"  [OK] All {len(boxes)} boxes are spatially distinct")

        # check 2: no two items resolved to identical top-1 result by coincidence of a bug
        # (same catalogue_id across items is FINE if genuinely similar; only flag identical
        #  full top-5 list, which would indicate the crops were accidentally identical)
        top5_lists = [tuple(r["catalogue_id"] for r in item["results"]) for item in items]
        for i in range(len(top5_lists)):
            for j in range(i + 1, len(top5_lists)):
                if top5_lists[i] == top5_lists[j]:
                    print(f"  [WARN] Item {i} and Item {j} have IDENTICAL top-5 — check for duplicate crop bug")

        for item in items:
            d = item["decision"]
            print(f"  Item {item['item_index']}: {item['detector_category']} -> "
                  f"{d['status'].upper()} (sim={d['similarity']:.3f})")

    print(f"\n{'='*60}")
    print(f"Spatial-distinctness check: {total_passed}/{total_checks} images passed cleanly")


if __name__ == "__main__":
    main()