"""
Phase 6 — cache query features once so every embedding variant (and Phase 7) is scored
by matrix products instead of re-embedding.

Queries = the Phase 1 same-listing queries of a split (fixed query image per listing).
  clean    : all eligible queries — features read from the stored catalogue embeddings
             (the query image IS a catalogue image), no compute
  <aug>    : the fixed 300-query subset under each Phase 1 augmentation (+ combined),
             re-embedded: DINOv2 CLS (raw), CLIP image embedding, and optionally DINOv2 of
             the background-removed image (--nobg)

Usage: python src/eval/query_cache.py --split val [--nobg]
  -> artifacts/eval_cache/<split>_<aug>.npz   (dino, clip[, nobg], query_index)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys

import numpy as np

sys.path.append(".")
from src.eval.common import Progress, open_image
from src.eval.augment import ALL_AUGS, AUG_MAX_SIDE, apply_aug
from src.eval.retrieval_eval import Ctx, embed_batch_raw
from src.catalogue.build_clip_embeddings import clip_image_embed

OUT = "artifacts/eval_cache"
BATCH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-sub", type=int, default=300)
    ap.add_argument("--nobg", action="store_true", help="also cache DINOv2 of background-removed queries")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx = Ctx(args.split)
    subset = ctx.subset(args.n_sub)
    qpos = {q["cid"]: i for i, q in enumerate(ctx.queries)}
    if args.nobg:
        from src.catalogue.build_nobg_embeddings import remove_bg

    for aug in [a for a in ALL_AUGS if a != "clean"]:
        path = os.path.join(OUT, f"{args.split}_{aug}.npz")
        have = dict(np.load(path)) if os.path.exists(path) else {}
        need_main = "dino" not in have
        need_nobg = args.nobg and "nobg" not in have
        if not (need_main or need_nobg):
            continue
        dino, clip, nobg = [], [], []
        prog = Progress(len(subset), f"{args.split}:{aug}")
        for b in range(0, len(subset), BATCH):
            batch = subset[b:b + BATCH]
            imgs = [apply_aug(aug, open_image(q["path"], AUG_MAX_SIDE), q["cid"]) for q in batch]
            if need_main:
                dino.append(embed_batch_raw(imgs))
                clip.append(clip_image_embed(imgs))
            if need_nobg:
                nobg.append(embed_batch_raw([remove_bg(im) for im in imgs]))
            prog.update(min(b + BATCH, len(subset)))
        out = dict(have)
        out["query_index"] = np.array([qpos[q["cid"]] for q in subset])
        if need_main:
            out["dino"], out["clip"] = np.concatenate(dino).astype(np.float32), np.concatenate(clip).astype(np.float32)
        if need_nobg:
            out["nobg"] = np.concatenate(nobg).astype(np.float32)
        np.savez_compressed(path, **out)
    print(f"Wrote {OUT}/{args.split}_*.npz")


if __name__ == "__main__":
    main()
