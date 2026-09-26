"""
Re-applies the tightened category classifier + exclude check to the
existing catalogue_metadata.parquet. Drops rows that no longer pass
(e.g. watches that slipped through the old loose keyword match) and
deletes their downloaded image files too.
"""

import os
import re
import shutil
import pandas as pd
import sys
sys.path.append(".")
import config


def passes_filter(row):
    text = " ".join([str(row.get("title", "")), str(row.get("category", ""))]).lower()

    if any(kw in text for kw in config.EXCLUDE_KEYWORDS):
        return False

    category = row["category"]
    keywords = config.CATEGORY_KEYWORDS.get(category, [])
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)


def main():
    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    print(f"Before cleanup: {len(df)} rows")

    mask = df.apply(passes_filter, axis=1)
    dropped = df[~mask]
    kept = df[mask]

    # remove downloaded images for dropped rows
    for _, row in dropped.iterrows():
        for path in row["image_paths"].split(";"):
            item_dir = os.path.dirname(path)
            if os.path.isdir(item_dir):
                shutil.rmtree(item_dir, ignore_errors=True)

    kept.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)

    print(f"Dropped: {len(dropped)} rows (likely watches / mismatches)")
    print(f"After cleanup: {len(kept)} rows")
    print("\nDropped examples:")
    print(dropped[["title", "category"]].head(10))
    print("\nRemaining category distribution:")
    print(kept["category"].value_counts())


if __name__ == "__main__":
    main()