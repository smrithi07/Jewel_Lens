"""
Phase 1 — perceptual hash (pHash, 64-bit) for every indexed catalogue
image, cached once so the retrieval harness can treat cross-listing
near-duplicates (Hamming <= 6) as relevant in its caveat variant.

Usage: python src/eval/phash_cache.py   -> data/phash_cache.parquet (~3 min)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

import imagehash
import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import PHASH_CACHE_PATH, Progress, open_image


def main():
    meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    hashes = np.zeros(len(meta), dtype=np.uint64)
    prog = Progress(len(meta), "phash")
    for i, p in enumerate(meta["image_path"]):
        h = imagehash.phash(open_image(p, 512))
        hashes[i] = np.uint64(int(str(h), 16))
        prog.update(i + 1)
    pd.DataFrame({"faiss_id": np.arange(len(meta)), "image_path": meta["image_path"],
                  "catalogue_id": meta["catalogue_id"], "phash": hashes}).to_parquet(PHASH_CACHE_PATH, index=False)
    print(f"Wrote {PHASH_CACHE_PATH} ({len(meta)} hashes)")


if __name__ == "__main__":
    main()
