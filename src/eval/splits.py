"""
Phase 1 — fixed evaluation splits, created once and reused by every phase.

1. data/splits.json — listing-level 70/15/15 train/val/test, stratified by
   category, seed 42. Per listing: split, category, n_images, eligible
   (>=2 indexed images, so a query image can be held out and the listing
   still has another image in the index), is_set (multi-piece set/combo by
   title) and a fixed query_image.
2. data/stumper/split.json — stumper photos split 50/50 tune/holdout by
   photo, stratified by folder, seed 42. Existing assignments are frozen;
   photos added later (e.g. Phase 3 negatives) are assigned to whichever
   half of their folder is smaller, so the original holdout never changes.
3. results/is_set_review.csv — every listing flagged is_set, for skimming.

Usage: python src/eval/splits.py            (refuses to overwrite data/splits.json)
       python src/eval/splits.py --stumper-only   (just assign new stumper photos)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(".")
import config
from src.eval.common import (SPLITS_PATH, STUMPER_SPLIT_PATH, STUMPER_ROOT, EXTRA_NEGATIVES_DIR,
                             IMAGE_EXTS, save_json, load_json)

SEED = 42

# Multi-piece listings: "Necklace Set", "Combo of 3", "Pack of 2", "Set of 3 Bracelets", "6 pcs".
SET_PATTERN = re.compile(
    r"\b(sets?|combos?|pack of|pair of [2-9]|duo|trio)\b|\b\d+\s*(pcs|pieces)\b",
    flags=re.IGNORECASE,
)


def build_catalogue_split():
    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    emb_meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    images_by_listing = emb_meta.groupby("catalogue_id")["image_path"].apply(list).to_dict()

    meta = meta.sort_values("catalogue_id").reset_index(drop=True)
    ids = meta["catalogue_id"].to_numpy()
    cats = meta["category"].to_numpy(dtype=str)

    train_ids, temp_ids, _, temp_cats = train_test_split(
        ids, cats, test_size=0.30, stratify=cats, random_state=SEED)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, stratify=temp_cats, random_state=SEED)
    split_of = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids},
                **{i: "test" for i in test_ids}}

    rng = np.random.default_rng(SEED)
    listings = {}
    for _, row in meta.iterrows():
        cid = row["catalogue_id"]
        imgs = sorted(images_by_listing.get(cid, []))
        eligible = len(imgs) >= 2
        listings[cid] = {
            "split": split_of[cid],
            "category": row["category"],
            "n_images": len(imgs),
            "eligible": eligible,
            "is_set": bool(SET_PATTERN.search(str(row["title"]))),
            "query_image": imgs[int(rng.integers(len(imgs)))] if eligible else None,
        }

    df = pd.DataFrame.from_dict(listings, orient="index")
    summary = {}
    for split in ("train", "val", "test"):
        d = df[df["split"] == split]
        summary[split] = {
            "listings": int(len(d)),
            "eligible_queries": int(d["eligible"].sum()),
            "eligible_non_set_queries": int((d["eligible"] & ~d["is_set"]).sum()),
            "is_set": int(d["is_set"].sum()),
            "by_category": d["category"].value_counts().to_dict(),
        }

    out = {"meta": {"seed": SEED, "created": str(date.today()), "ratios": [0.70, 0.15, 0.15],
                    "stratify": "category", "summary": summary},
           "listings": listings}
    save_json(out, SPLITS_PATH)

    review = meta[meta["catalogue_id"].isin(df.index[df["is_set"]])][["catalogue_id", "category", "title"]]
    review.to_csv("results/is_set_review.csv", index=False)
    return out, len(review)


def list_stumper_photos():
    """{photo_key: folder}; photo_key is the path relative to the repo root."""
    photos = {}
    for folder in sorted(os.listdir(STUMPER_ROOT)):
        d = os.path.join(STUMPER_ROOT, folder)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                photos[os.path.join(d, f)] = folder
    if os.path.isdir(EXTRA_NEGATIVES_DIR):
        for f in sorted(os.listdir(EXTRA_NEGATIVES_DIR)):
            if f.lower().endswith(IMAGE_EXTS):
                photos[os.path.join(EXTRA_NEGATIVES_DIR, f)] = "negatives"
    return photos


def build_stumper_split():
    photos = list_stumper_photos()
    existing = load_json(STUMPER_SPLIT_PATH)["photos"] if os.path.exists(STUMPER_SPLIT_PATH) else {}

    assigned = {k: v for k, v in existing.items() if k in photos}
    rng = np.random.default_rng(SEED)
    for folder in sorted(set(photos.values())):
        new = [p for p, f in photos.items() if f == folder and p not in assigned]
        if not new:
            continue
        rng.shuffle(new)
        for p in new:
            n_hold = sum(1 for k, v in assigned.items() if photos.get(k) == folder and v["half"] == "holdout")
            n_tune = sum(1 for k, v in assigned.items() if photos.get(k) == folder and v["half"] == "tune")
            half = "holdout" if n_hold <= n_tune else "tune"
            assigned[p] = {"folder": folder, "half": half, "added": str(date.today())}

    summary = {}
    for k, v in assigned.items():
        s = summary.setdefault(v["folder"], {"tune": 0, "holdout": 0})
        s[v["half"]] += 1
    save_json({"meta": {"seed": SEED, "stratify": "folder",
                        "rule": "existing assignments frozen; new photos go to the smaller half of their folder",
                        "summary": summary},
               "photos": dict(sorted(assigned.items()))}, STUMPER_SPLIT_PATH)
    return summary, len(assigned) - len(existing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stumper-only", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    if not args.stumper_only:
        if os.path.exists(SPLITS_PATH):
            sys.exit(f"{SPLITS_PATH} already exists — splits are fixed once. Delete it by hand only if you mean to reset every phase.")
        out, n_set = build_catalogue_split()
        print(f"Wrote {SPLITS_PATH}")
        for split, s in out["meta"]["summary"].items():
            print(f"  {split:5s} listings={s['listings']:4d} eligible={s['eligible_queries']:4d} "
                  f"eligible_non_set={s['eligible_non_set_queries']:4d} is_set={s['is_set']:3d} {s['by_category']}")
        print(f"  is_set flagged: {n_set} listings -> results/is_set_review.csv")

    summary, n_new = build_stumper_split()
    print(f"Wrote {STUMPER_SPLIT_PATH} ({n_new} newly assigned photos)")
    for folder, s in summary.items():
        print(f"  {folder:10s} tune={s['tune']:3d} holdout={s['holdout']:3d}")


if __name__ == "__main__":
    main()
