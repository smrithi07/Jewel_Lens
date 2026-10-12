"""
Phase 4 — per-item inputs for every category signal, on the stumper photos.
Uses the SAME boxes as the baseline (data/eval_own/baseline_detections.json),
so Phase 4 measures categorisation only; detection changes are Phase 5.

Per item it stores: DINOv2 CLS of the crop (raw), CLIP image embedding of the
crop, the detector label/score, and the top-20 retrieved listings (category,
score) from the production index. Signals are computed from these in
src/eval/category_fusion.py, so trying a new k / prompt / weight is instant.

Usage: python src/eval/category_signals.py  -> data/eval_own/phase4_items.npz (~2 min)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys

import numpy as np

sys.path.append(".")
from src.eval.common import Progress, load_json
from src.eval.retrieval_eval import embed_batch_raw
from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.detect import crop_with_padding
from src.pipeline.retrieve import search
from src.catalogue.build_clip_embeddings import clip_image_embed

DETS = "data/eval_own/baseline_detections.json"
OUT = "data/eval_own/phase4_items.npz"


def main():
    dets = load_json(DETS)
    rows, dino, clip, knn = [], [], [], []
    prog = Progress(len(dets), "items")
    for i, (photo, d) in enumerate(dets.items()):
        if d["items"]:
            img = load_and_preprocess(photo)
            crops = [crop_with_padding(img, it["box"]) for it in d["items"]]
            raws = embed_batch_raw(crops)
            dino.append(raws)
            clip.append(clip_image_embed(crops))
            for it, raw in zip(d["items"], raws):
                res = search(raw / np.linalg.norm(raw), top_k=20)
                knn.append([(r["category"], r["score"], r["catalogue_id"]) for r in res])
                rows.append({"photo": photo, "folder": d["folder"], "half": d["half"], "item_index": it["item_index"],
                             "detector_category": it["detector_category"], "detector_score": it["detector_score"]})
        prog.update(i + 1)
    np.savez_compressed(OUT, dino=np.concatenate(dino).astype(np.float32), clip=np.concatenate(clip).astype(np.float32),
                        rows=np.array(json.dumps(rows)), knn=np.array(json.dumps(knn)))
    print(f"Wrote {OUT}: {len(rows)} items")


if __name__ == "__main__":
    main()
