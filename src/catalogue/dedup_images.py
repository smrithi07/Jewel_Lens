"""
Phase 1.5a — perceptual-hash dedup across the whole catalogue.
Drops near-identical images (same stock photo reused across stores/
listings), keeping one representative per duplicate cluster.
"""

import imagehash
from PIL import Image
import pandas as pd
import sys
sys.path.append(".")
import config


def compute_hashes(df):
    records = []
    for _, row in df.iterrows():
        for img_path in row["image_paths"].split(";"):
            try:
                h = imagehash.phash(Image.open(img_path))
                records.append({"image_path": img_path, "catalogue_id": row["catalogue_id"], "hash": h})
            except Exception as e:
                print(f"  [warn] unreadable {img_path}: {e}")
    return records


def find_duplicate_clusters(records, threshold):
    """Greedy clustering by Hamming distance. Returns list of clusters (lists of records)."""
    remaining = records.copy()
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still_remaining = []
        for r in remaining:
            if seed["hash"] - r["hash"] <= threshold:
                cluster.append(r)
            else:
                still_remaining.append(r)
        remaining = still_remaining
        clusters.append(cluster)
    return clusters


def main():
    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    records = compute_hashes(df)
    print(f"Hashed {len(records)} images")

    clusters = find_duplicate_clusters(records, config.PHASH_THRESHOLD)
    dup_clusters = [c for c in clusters if len(c) > 1]
    print(f"Found {len(dup_clusters)} duplicate clusters, {sum(len(c)-1 for c in dup_clusters)} redundant images")

    keep_paths = set()
    drop_paths = set()
    log_rows = []
    for cluster in clusters:
        keeper = cluster[0]
        keep_paths.add(keeper["image_path"])
        for dup in cluster[1:]:
            drop_paths.add(dup["image_path"])
            log_rows.append({"dropped": dup["image_path"], "kept": keeper["image_path"],
                              "dropped_catalogue_id": dup["catalogue_id"], "kept_catalogue_id": keeper["catalogue_id"]})

    pd.DataFrame(log_rows).to_csv(config.DEDUP_LOG_PATH, index=False)

    def filter_paths(image_paths_str):
        kept = [p for p in image_paths_str.split(";") if p not in drop_paths]
        return ";".join(kept)

    df["image_paths"] = df["image_paths"].apply(filter_paths)
    df = df[df["image_paths"] != ""]  # drop rows with zero images left
    df.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)

    total_images = df["image_paths"].apply(lambda p: len(p.split(";"))).sum()
    print(f"\nAfter dedup: {len(df)} listings, {total_images} images")
    print(f"Dedup log written to {config.DEDUP_LOG_PATH}")


if __name__ == "__main__":
    main()