"""
Phase 4 — training/eval features for the retrained category classifier.
Catalogue images only (never stumper photos). Set listings are excluded
(multi-piece, ambiguous category). Labels = listing category.

Per split (train, val):
  clean    : stored DINOv2 embedding of every image of every non-set listing
  aug      : one random phone-photo augmentation per image (src/eval/augment.py)
  detcrop  : OWL-ViT detector crops (top 2 boxes) of a category-stratified
             subset of images (train ~2000, val ~400) — what the classifier
             actually sees at inference

Saves raw (un-normalised) CLS vectors; the trainer decides the normalisation.

Usage: python src/catalogue/build_category_data.py   (~35-45 min on CPU, prints ETA)
  -> artifacts/category_v2/features_{train,val}.npz
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import SPLITS_PATH, Progress, load_json, open_image
from src.eval.augment import AUGMENTATIONS, apply_aug, AUG_MAX_SIDE
from src.eval.retrieval_eval import embed_batch_raw
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.preprocess import preprocess_image

OUT_DIR = "artifacts/category_v2"
SEED = 42
N_DETCROP = {"train": 2000, "val": 400}
BATCH = 16


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    listings = load_json(SPLITS_PATH)["listings"]
    meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    raw = np.load(config.CATALOGUE_EMBEDDINGS_PATH)
    meta["split"] = meta["catalogue_id"].map(lambda c: listings[c]["split"])
    meta["is_set"] = meta["catalogue_id"].map(lambda c: listings[c]["is_set"])
    rng = np.random.default_rng(SEED)

    for split in ("train", "val"):
        out_path = os.path.join(OUT_DIR, f"features_{split}.npz")
        if os.path.exists(out_path):
            print(f"{out_path} exists, skipping")
            continue
        m = meta[(meta["split"] == split) & ~meta["is_set"]]
        X, y, cid, src, aug_name = [raw[m.index]], list(m["category"]), list(m["catalogue_id"]), ["clean"] * len(m), [""] * len(m)

        # one augmented view per image (cached, so a later failure does not redo this stage)
        paths = list(m["image_path"])
        augs = [AUGMENTATIONS[i] for i in rng.integers(len(AUGMENTATIONS), size=len(paths))]
        aug_cache = os.path.join(OUT_DIR, f"_aug_{split}.npy")
        if os.path.exists(aug_cache):
            aug_X = np.load(aug_cache)
        else:
            prog = Progress(len(paths), f"{split}:aug")
            feats = []
            for b in range(0, len(paths), BATCH):
                imgs = [apply_aug(a, open_image(p, AUG_MAX_SIDE), f"train-cat|{p}") for p, a in zip(paths[b:b + BATCH], augs[b:b + BATCH])]
                feats.append(embed_batch_raw(imgs))
                prog.update(min(b + BATCH, len(paths)))
            aug_X = np.concatenate(feats)
            np.save(aug_cache, aug_X)
        X.append(aug_X); y += list(m["category"]); cid += list(m["catalogue_id"])
        src += ["aug"] * len(m); aug_name += augs

        # detector crops on a category-stratified subset
        load_detector()
        n = N_DETCROP[split]
        sub = pd.concat([g.sample(n=min(len(g), round(n * len(g) / len(m))), random_state=SEED)
                         for _, g in m.groupby("category")])
        prog = Progress(len(sub), f"{split}:detcrop")
        crops, crop_y, crop_cid = [], [], []
        for i, (_, r) in enumerate(sub.iterrows()):
            img = preprocess_image(open_image(r["image_path"]))
            for d in sorted(detect(img), key=lambda d: -d["score"])[:2]:
                crops.append(crop_with_padding(img, d["box"])); crop_y.append(r["category"]); crop_cid.append(r["catalogue_id"])
            if len(crops) >= BATCH:
                X.append(embed_batch_raw(crops)); y += crop_y; cid += crop_cid
                src += ["detcrop"] * len(crops); aug_name += [""] * len(crops)
                crops, crop_y, crop_cid = [], [], []
            prog.update(i + 1)
        if crops:
            X.append(embed_batch_raw(crops)); y += crop_y; cid += crop_cid
            src += ["detcrop"] * len(crops); aug_name += [""] * len(crops)

        X = np.concatenate(X).astype(np.float32)
        np.savez_compressed(out_path, X=X, y=np.array(y), catalogue_id=np.array(cid), source=np.array(src), aug=np.array(aug_name))
        print(f"Wrote {out_path}: {X.shape}, sources {pd.Series(src).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
