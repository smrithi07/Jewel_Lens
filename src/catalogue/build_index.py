"""
Phase 2b — build a single flat FAISS HNSW index (cosine similarity via
normalized inner product) over the embeddings from build_embeddings.py.
Saves the index + a faiss_id -> catalogue metadata mapping.
"""

import time
import faiss
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
import config


def main():
    print("Loading embeddings...")
    embeddings = np.load(config.CATALOGUE_EMBEDDINGS_PATH).astype("float32")
    meta_df = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    print(f"  {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")

    print("Normalizing embeddings (L2, for cosine similarity)...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"Building HNSW index (M=32, efConstruction=200)...")
    t0 = time.time()
    base_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    base_index.hnsw.efConstruction = 200
    base_index.hnsw.efSearch = 64
    index = faiss.IndexIDMap(base_index)

    ids = np.arange(len(embeddings)).astype("int64")

    print("Adding vectors to index...")
    add_batch_size = 2000
    for start in range(0, len(embeddings), add_batch_size):
        end = min(start + add_batch_size, len(embeddings))
        index.add_with_ids(embeddings[start:end], ids[start:end])
        print(f"  added {end}/{len(embeddings)} vectors")

    build_time = time.time() - t0
    print(f"Index build complete in {build_time:.1f}s")

    faiss.write_index(index, config.FAISS_INDEX_PATH)
    print(f"Saved index to: {config.FAISS_INDEX_PATH}")

    print("Building faiss_id -> catalogue metadata mapping...")
    id_mapping = {
        int(faiss_id): {
            "catalogue_id": meta_df.iloc[i]["catalogue_id"],
            "category": meta_df.iloc[i]["category"],
            "image_path": meta_df.iloc[i]["image_path"],
        }
        for i, faiss_id in enumerate(ids)
    }
    with open(config.ID_MAPPING_PATH, "wb") as f:
        pickle.dump(id_mapping, f)
    print(f"Saved mapping to: {config.ID_MAPPING_PATH}")

    print(f"\n{'='*50}")
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    print("\nRunning self-query checkpoint...")
    query = embeddings[0:1]
    scores, result_ids = index.search(query, k=3)
    top1_catalogue_id = id_mapping[int(result_ids[0][0])]["catalogue_id"]
    expected_catalogue_id = meta_df.iloc[0]["catalogue_id"]
    print(f"  top1 catalogue_id = {top1_catalogue_id}")
    print(f"  expected          = {expected_catalogue_id}")
    print(f"  similarity score  = {scores[0][0]:.4f}")

    assert top1_catalogue_id == expected_catalogue_id, "Self-query failed — top1 should be the query item itself"
    assert scores[0][0] > 0.99, "Self-query similarity too low — check normalization"
    print("PASS")


if __name__ == "__main__":
    main()