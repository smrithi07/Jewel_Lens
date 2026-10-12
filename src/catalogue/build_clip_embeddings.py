"""
Phase 2 — CLIP (openai/clip-vit-base-patch32) image embeddings for every
indexed catalogue image, L2-normalised, in the SAME row order as
data/catalogue_embeddings_meta.parquet (row i <-> faiss id i).

Used by Phase 2 (zero-shot subtype/colour backfill), and reusable by
Phase 4 (CLIP crop verification) and Phase 6 (variant C / fused D).
Does not touch any production artifact.

Usage: python src/catalogue/build_clip_embeddings.py   (~10 min on CPU)
  -> artifacts/clip_b32/embeddings.npy, artifacts/clip_b32/meta.parquet
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.append(".")
import config
from src.eval.common import Progress, open_image

OUT_DIR = "artifacts/clip_b32"
BATCH = 32

_model = _processor = None


def load_clip():
    global _model, _processor
    if _model is None:
        _processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        _model = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
    return _model, _processor


def _features(out):
    # transformers 5 may return a ModelOutput instead of a bare tensor
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def clip_image_embed(images):
    model, processor = load_clip()
    with torch.no_grad():
        f = _features(model.get_image_features(**processor(images=images, return_tensors="pt")))
    f = f.numpy().astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def clip_text_embed(texts):
    model, processor = load_clip()
    with torch.no_grad():
        f = _features(model.get_text_features(**processor(text=texts, return_tensors="pt", padding=True)))
    f = f.numpy().astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    paths = meta["image_path"].tolist()
    out = np.zeros((len(paths), 512), dtype=np.float32)
    prog = Progress(len(paths), "clip")
    for b in range(0, len(paths), BATCH):
        imgs = [open_image(p, 448) for p in paths[b:b + BATCH]]
        out[b:b + len(imgs)] = clip_image_embed(imgs)
        prog.update(b + len(imgs))
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), out)
    meta.to_parquet(os.path.join(OUT_DIR, "meta.parquet"), index=False)
    print(f"Wrote {OUT_DIR}/embeddings.npy {out.shape}")


if __name__ == "__main__":
    main()
