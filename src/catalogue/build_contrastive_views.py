"""
Phase 7 — precompute frozen DINOv2 embeddings of several views of every TRAIN-split image,
so the projection head can be trained cheaply on CPU. Train split only (never val/test
listings, never stumper photos).

Views per image
  clean : the stored catalogue embedding (no compute)
  aug1  : the Phase 4 augmented view (reused from artifacts/category_v2/_aug_train.npy,
          non-set images only)
  aug2.. : K new random views; each is one Phase 1 augmentation or 'combined' (3 stacked),
          with 'combined', 'busy_bg' and 'blur' over-weighted because those are the weakest
          conditions (Phase 6)

Usage: python src/catalogue/build_contrastive_views.py [--k 3]   (resumable; ~45 min plugged in)
  -> artifacts/contrastive/views_train.npz : X (float16 raw CLS), fid, view
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import SPLITS_PATH, Progress, load_json, open_image
from src.eval.augment import AUGMENTATIONS, AUG_MAX_SIDE, apply_aug
from src.eval.retrieval_eval import embed_batch_raw

OUT = "artifacts/contrastive"
BATCH = 16
SEED = 7
AUG_POOL = AUGMENTATIONS + ["combined", "combined", "busy_bg", "blur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    L = load_json(SPLITS_PATH)["listings"]
    meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    split = meta["catalogue_id"].map(lambda c: L[c]["split"]).to_numpy()
    is_set = meta["catalogue_id"].map(lambda c: L[c]["is_set"]).to_numpy()
    raw = np.load(config.CATALOGUE_EMBEDDINGS_PATH)
    tr = np.where(split == "train")[0]

    X, fid, view = [raw[tr]], [tr], ["clean"] * len(tr)
    # reuse the Phase 4 augmented view (same row order as build_category_data: train & non-set)
    p4 = "artifacts/category_v2/_aug_train.npy"
    if os.path.exists(p4):
        nonset = np.where((split == "train") & ~is_set)[0]
        a = np.load(p4)
        assert len(a) == len(nonset)
        X.append(a); fid.append(nonset); view += ["aug1"] * len(nonset)

    rng = np.random.default_rng(SEED)
    for k in range(args.k):
        part = os.path.join(OUT, f"_view{k + 2}.npy")
        augs = [AUG_POOL[i] for i in rng.integers(len(AUG_POOL), size=len(tr))]   # drawn even if cached
        if os.path.exists(part):
            Xk = np.load(part)
        else:
            feats = []
            prog = Progress(len(tr), f"view{k + 2}")
            paths = meta["image_path"].to_numpy()[tr]
            for b in range(0, len(tr), BATCH):
                imgs = [apply_aug(a, open_image(p, AUG_MAX_SIDE), f"contrastive{k}|{p}")
                        for p, a in zip(paths[b:b + BATCH], augs[b:b + BATCH])]
                feats.append(embed_batch_raw(imgs).astype(np.float32))
                prog.update(min(b + BATCH, len(tr)))
            Xk = np.concatenate(feats)
            np.save(part, Xk)
        X.append(Xk); fid.append(tr); view += [f"aug{k + 2}:{a}" for a in augs]

    X = np.concatenate(X).astype(np.float16)
    np.savez_compressed(os.path.join(OUT, "views_train.npz"), X=X, fid=np.concatenate(fid), view=np.array(view))
    print(f"Wrote {OUT}/views_train.npz: {X.shape} ({len(tr)} train images)")


if __name__ == "__main__":
    main()
