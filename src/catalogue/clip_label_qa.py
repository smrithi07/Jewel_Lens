"""
Phase 1.5b — CLIP zero-shot label QA.
Checks each catalogue image against its stated category using CLIP
zero-shot classification. Confident mismatches are auto-dropped;
borderline ones are logged for manual review but kept.
"""

import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.append(".")
import config

NOT_JEWELRY_PROMPTS = [
    "a photo of clothing on a hanger",
    "a photo of a logo or icon",
    "a photo of packaging or a box",
]

BATCH_SIZE = 32


def load_model():
    model = CLIPModel.from_pretrained(config.CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model.eval()
    return model, processor


def build_prompts():
    category_prompts = [f"a photo of a {cat}" for cat in config.CATEGORIES]
    all_prompts = category_prompts + NOT_JEWELRY_PROMPTS
    labels = config.CATEGORIES + ["not_jewelry"] * len(NOT_JEWELRY_PROMPTS)
    return all_prompts, labels


def classify_batch(model, processor, image_paths, prompts):
    images = []
    valid_paths = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"  [warn] unreadable {p}: {e}")
    if not images:
        return [], []

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).numpy()  # (n_images, n_prompts)
    return probs, valid_paths


def collapse_to_category_scores(probs, labels):
    """Collapse the not_jewelry prompts into one summed score, keep categories separate."""
    results = []
    for row in probs:
        scores = {}
        for label, p in zip(labels, row):
            scores[label] = scores.get(label, 0.0) + float(p)
        results.append(scores)
    return results


def main():
    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    model, processor = load_model()
    prompts, labels = build_prompts()

    drop_paths = set()
    log_rows = []
    batch_paths, batch_meta = [], []

    def flush():
        if not batch_paths:
            return
        probs, valid_paths = classify_batch(model, processor, batch_paths, prompts)
        if len(probs) == 0:
            return
        scores_list = collapse_to_category_scores(probs, labels)
        valid_set = {p: m for p, m in zip(batch_paths, batch_meta) if p in valid_paths}

        for path, scores in zip(valid_paths, scores_list):
            meta = valid_set[path]
            stated_category = meta["category"]
            predicted_category = max(scores, key=scores.get)
            predicted_score = scores[predicted_category]
            stated_score = scores.get(stated_category, 0.0)

            if predicted_category != stated_category:
                confidence_gap = predicted_score - stated_score
                log_rows.append({
                    "image_path": path,
                    "catalogue_id": meta["catalogue_id"],
                    "stated_category": stated_category,
                    "predicted_category": predicted_category,
                    "predicted_score": round(predicted_score, 3),
                    "stated_score": round(stated_score, 3),
                    "action": "auto_dropped" if confidence_gap > config.CLIP_AUTO_DROP_THRESHOLD
                              else ("flagged" if confidence_gap > config.CLIP_FLAG_THRESHOLD else "kept"),
                })
                if confidence_gap > config.CLIP_AUTO_DROP_THRESHOLD:
                    drop_paths.add(path)

    for _, row in df.iterrows():
        for img_path in row["image_paths"].split(";"):
            batch_paths.append(img_path)
            batch_meta.append({"catalogue_id": row["catalogue_id"], "category": row["category"]})
            if len(batch_paths) >= BATCH_SIZE:
                flush()
                batch_paths, batch_meta = [], []
    flush()

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(config.CLIP_QA_LOG_PATH, index=False)

    def filter_paths(image_paths_str):
        kept = [p for p in image_paths_str.split(";") if p not in drop_paths]
        return ";".join(kept)

    df["image_paths"] = df["image_paths"].apply(filter_paths)
    df = df[df["image_paths"] != ""]
    df.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)

    total_images = df["image_paths"].apply(lambda p: len(p.split(";"))).sum()
    print(f"Auto-dropped: {len(drop_paths)} images")
    print(f"Flagged for review (kept): {(log_df['action'] == 'flagged').sum() if len(log_df) else 0}")
    print(f"\nAfter QA: {len(df)} listings, {total_images} images")
    print(f"Full mismatch log: {config.CLIP_QA_LOG_PATH}")


if __name__ == "__main__":
    main()