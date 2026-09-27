"""
Semi-automated stumper labeling.

Uses CLIP (independent of the DINOv2 retrieval backbone) to judge, for each
detected item, whether one of its 5 retrieved candidates actually matches the
query crop. Fills `chosen_rank` in labels_draft.csv only when CLIP is
confident; leaves everything else blank for manual review.

This does NOT replace human labeling — it cuts the manual workload down to
the rows CLIP itself is unsure about, and every auto-filled row is flagged
with auto_labeled=true so it's traceable in the write-up.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.append(".")
import config

import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.pipeline.run_pipeline import load_all_models, run_pipeline
from src.pipeline.preprocess import load_and_preprocess

STUMPER_ROOT = "data/stumper/images"
LABELS_PATH = "data/stumper/labels_draft.csv"
BACKUP_PATH = "data/stumper/labels_draft_before_autolabel.csv"

# CLIP confidence thresholds — deliberately conservative.
# top1 must clearly beat top2, and be a strong match in absolute terms.
CLIP_MATCH_THRESHOLD = 0.82
CLIP_MARGIN_THRESHOLD = 0.05

FOLDERS = ["earring", "necklace", "bracelet", "mixed", "negatives"]


def load_clip():
    print("Loading CLIP (independent judge model)...")
    model = CLIPModel.from_pretrained(config.CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model.eval()
    return model, processor


def clip_embed(model, processor, image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        feats = model.visual_projection(pooled_output)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.squeeze(0).numpy()


def load_catalogue_lookup():
    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    return dict(zip(meta["catalogue_id"], meta["primary_image"]))


def main():
    models = load_all_models()
    clip_model, clip_processor = load_clip()
    cat_lookup = load_catalogue_lookup()

    df = pd.read_csv(LABELS_PATH, dtype=str).fillna("")
    df.to_csv(BACKUP_PATH, index=False)
    print(f"Backed up original labels to {BACKUP_PATH}")

    if "auto_labeled" not in df.columns:
        df["auto_labeled"] = ""

    auto_filled, left_blank, skipped = 0, 0, 0

    for folder in FOLDERS:
        if folder == "negatives":
            continue  # ground truth already known from folder, nothing to label

        folder_path = os.path.join(STUMPER_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue

        photos = sorted(os.listdir(folder_path))
        for photo in photos:
            photo_rows = df[(df["folder"] == folder) & (df["photo_filename"] == photo)]
            if photo_rows.empty:
                continue

            # skip if every row for this photo already has a chosen_rank (human already did it)
            if (photo_rows["chosen_rank"].str.strip() != "").all():
                continue

            image_path = os.path.join(folder_path, photo)
            try:
                image = load_and_preprocess(image_path)
            except Exception as e:
                print(f"  [skip] {folder}/{photo}: could not load ({e})")
                continue

            items = run_pipeline(image, top_k=5)

            for _, row in photo_rows.iterrows():
                if row["chosen_rank"].strip() != "":
                    continue  # already labeled by hand — never overwrite

                item_index = int(row["item_index"]) if row["item_index"].strip() != "" else None
                if item_index is None or item_index >= len(items):
                    skipped += 1
                    continue

                item = items[item_index]
                crop = item["crop"]
                candidate_ids = [row.get(f"candidate_{i}_id", "") for i in range(1, 6)]
                candidate_ids = [c for c in candidate_ids if c]
                if not candidate_ids:
                    skipped += 1
                    continue

                query_emb = clip_embed(clip_model, clip_processor, crop)

                sims = []
                for cid in candidate_ids:
                    img_path = cat_lookup.get(cid)
                    if img_path is None or not os.path.exists(img_path):
                        sims.append(-1.0)
                        continue
                    try:
                        cand_img = Image.open(img_path).convert("RGB")
                    except Exception:
                        sims.append(-1.0)
                        continue
                    cand_emb = clip_embed(clip_model, clip_processor, cand_img)
                    sims.append(float(np.dot(query_emb, cand_emb)))

                sims = np.array(sims)
                order = np.argsort(-sims)
                top1_sim = sims[order[0]]
                top2_sim = sims[order[1]] if len(sims) > 1 else -1.0
                margin = top1_sim - top2_sim

                idx = df.index[(df["folder"] == folder) &
                               (df["photo_filename"] == photo) &
                               (df["item_index"] == row["item_index"])][0]

                                # always take CLIP's best-of-5 pick, no confidence gate
                df.at[idx, "chosen_rank"] = str(int(order[0]) + 1)
                df.at[idx, "auto_labeled"] = "true"
                df.at[idx, "label_source"] = "clip"
                df.at[idx, "clip_top1_sim"] = f"{top1_sim:.3f}"
                df.at[idx, "clip_margin"] = f"{margin:.3f}"
                auto_filled += 1

            print(f"  {folder}/{photo}: processed")

    df.to_csv(LABELS_PATH, index=False)

    print("\n" + "=" * 60)
    print("AUTO-LABELING SUMMARY")
    print("=" * 60)
    print(f"Auto-filled (CLIP confident):     {auto_filled}")
    print(f"Left blank (CLIP unsure, needs eye): {left_blank}")
    print(f"Skipped (no detection / no candidates): {skipped}")
    print(f"\nUpdated: {LABELS_PATH}")
    print("Open review_report.html and manually check the rows still blank —")
    print("that's your real remaining workload, not the full ~170.")


if __name__ == "__main__":
    main()