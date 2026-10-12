"""
Phase 6 variant B — DINOv2 embeddings of background-removed catalogue images.
rembg/u2net cut-out composited onto white (the usual catalogue look), then the same
DINOv2 CLS embedding as production. Row order = data/catalogue_embeddings_meta.parquet.
Resumable (partial results saved every 500 images).

Usage: python src/catalogue/build_nobg_embeddings.py   (~1.5 h CPU, plugged in)
  -> artifacts/dinov2_nobg/embeddings.npy (raw CLS), meta.parquet
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(".")
import config
from src.eval.common import Progress, open_image
from src.eval.retrieval_eval import embed_batch_raw

OUT = "artifacts/dinov2_nobg"
BATCH = 16
_session = None


def remove_bg(img):
    """PIL RGB -> RGB with the background replaced by white (u2net cut-out)."""
    global _session
    import rembg
    if _session is None:
        _session = rembg.new_session("u2net")
    cut = rembg.remove(img, session=_session)
    white = Image.new("RGB", cut.size, (255, 255, 255))
    white.paste(cut, mask=cut.split()[-1])
    return white


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    part = os.path.join(OUT, "_partial.npy")
    done = np.load(part) if os.path.exists(part) else np.zeros((0, 768), np.float32)
    paths = meta["image_path"].tolist()
    feats = [done]
    prog = Progress(len(paths), "nobg")
    start = len(done)
    for b in range(start, len(paths), BATCH):
        imgs = [remove_bg(open_image(p, 1024)) for p in paths[b:b + BATCH]]
        feats.append(embed_batch_raw(imgs).astype(np.float32))
        n = b + len(imgs)
        if n % 512 < BATCH or n == len(paths):
            np.save(part, np.concatenate(feats))
        prog.update(n)
    X = np.concatenate(feats)
    np.save(os.path.join(OUT, "embeddings.npy"), X)
    meta.to_parquet(os.path.join(OUT, "meta.parquet"), index=False)
    os.remove(part)
    print(f"Wrote {OUT}/embeddings.npy {X.shape}")


if __name__ == "__main__":
    main()
