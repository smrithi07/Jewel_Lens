"""
Throwaway Phase 3 checkpoint script — visualizes OWL-ViT detections on
a handful of test photos. Put a few test images in data/detect_test/
before running. Re-running overwrites both the annotated image and the
logged detections for any filename already tested — old runs never pile up.
"""

import os
import json
from PIL import Image, ImageDraw
import sys
sys.path.append(".")
from src.pipeline.detect import detect, load_detector

TEST_DIR = "data/detect_test"
OUTPUT_DIR = "data/detect_test_output"
LOG_PATH = "data/detect_test_output/detections_log.json"


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_detector()
    log = load_log()

    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not test_images:
        print(f"No test images found in {TEST_DIR} — add a few photos first.")
        return

    for fname in test_images:
        path = os.path.join(TEST_DIR, fname)
        image = Image.open(path).convert("RGB")
        detections = detect(image)

        print(f"\n{fname}: {len(detections)} detection(s)")
        for d in detections:
            print(f"  {d['category']} — score {d['score']:.3f} — box {tuple(round(x) for x in d['box'])}")

        # overwrite this filename's entry in the log (not append)
        log[fname] = detections

        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        for d in detections:
            draw.rectangle(d["box"], outline="red", width=3)
            draw.text((d["box"][0], d["box"][1] - 12), f"{d['category']} {d['score']:.2f}", fill="red")
        draw_img.save(os.path.join(OUTPUT_DIR, f"annotated_{fname}"))  # same filename = overwrite

    save_log(log)
    print(f"\nAnnotated images saved to {OUTPUT_DIR}/")
    print(f"Detection log (overwritten per-image) saved to {LOG_PATH}")


if __name__ == "__main__":
    main()