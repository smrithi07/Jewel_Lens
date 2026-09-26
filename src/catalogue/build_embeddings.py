"""
Phase 2a — embed every catalogue image (all images per listing) with DINOv2.
Saves embeddings.npy (N x dim) + meta.parquet (image_path, catalogue_id, category)
in matching row order. Prints progress every batch so long CPU runs are visible.
"""

import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.append(".")
import config

BATCH_SIZE = 16


def load_model():
    print(f"Loading embedder: {config.EMBEDDER_MODEL} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(config.EMBEDDER_MODEL)
    model = AutoModel.from_pretrained(config.EMBEDDER_MODEL)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return processor, model


def embed_batch(processor, model, image_paths):
    images = []
    valid_paths = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"    [warn] skipping unreadable image {p}: {e}")
    if not images:
        return np.empty((0, 768)), []

    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings, valid_paths


def main():
    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    total_images = df["image_paths"].apply(lambda p: len(p.split(";"))).sum()
    print(f"Catalogue: {len(df)} listings, {total_images} images to embed\n")

    processor, model = load_model()

    all_embeddings = []
    meta_rows = []
    batch_paths, batch_meta = [], []
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    batch_num = 0
    total_batches = -(-total_images // BATCH_SIZE)  # ceil division

    def flush():
        nonlocal processed_count, skipped_count, batch_num
        if not batch_paths:
            return
        batch_num += 1
        emb, valid_paths = embed_batch(processor, model, batch_paths)
        valid_set = set(valid_paths)
        kept_meta = [m for p, m in zip(batch_paths, batch_meta) if p in valid_set]
        all_embeddings.append(emb)
        meta_rows.extend(kept_meta)

        processed_count += len(valid_paths)
        skipped_count += len(batch_paths) - len(valid_paths)

        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        remaining = total_images - processed_count
        eta_sec = remaining / rate if rate > 0 else 0

        print(f"  batch {batch_num}/{total_batches} | "
              f"embedded {processed_count}/{total_images} "
              f"({100 * processed_count / total_images:.1f}%) | "
              f"{rate:.1f} img/s | ETA {eta_sec / 60:.1f} min")

    for _, row in df.iterrows():
        for img_path in row["image_paths"].split(";"):
            batch_paths.append(img_path)
            batch_meta.append({
                "image_path": img_path,
                "catalogue_id": row["catalogue_id"],
                "category": row["category"],
            })
            if len(batch_paths) >= BATCH_SIZE:
                flush()
                batch_paths, batch_meta = [], []
    flush()

    embeddings = np.concatenate(all_embeddings, axis=0)
    meta_df = pd.DataFrame(meta_rows)

    np.save(config.CATALOGUE_EMBEDDINGS_PATH, embeddings)
    meta_df.to_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH, index=False)

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"DONE in {total_time / 60:.1f} min")
    print(f"Embedded: {len(embeddings)} images (skipped {skipped_count} unreadable)")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Unique catalogue items covered: {meta_df['catalogue_id'].nunique()}")
    print(f"Saved to: {config.CATALOGUE_EMBEDDINGS_PATH}")
    print(f"Saved to: {config.CATALOGUE_EMBEDDINGS_META_PATH}")


if __name__ == "__main__":
    main()