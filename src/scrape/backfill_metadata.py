"""
Backfills subtype/color_query on an existing catalogue_metadata.parquet
without re-downloading any images. Re-fetches products.json pages (cheap)
and joins by catalogue_id.
"""

import time
import requests
import pandas as pd
import sys
sys.path.append(".")
import config
from src.scrape.shopify_scrapper import extract_color, extract_subtype, classify_category


def refetch_products(domain, max_pages=50):
    products = []
    page = 1
    while page <= max_pages:
        resp = requests.get(f"https://{domain}/products.json", params={"limit": 250, "page": page}, timeout=15)
        if resp.status_code != 200:
            break
        batch = resp.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        page += 1
        time.sleep(0.5)
    return products


def main():
    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    updated = 0

    for domain in config.SHOPIFY_STORE_DOMAINS:
        print(f"Refetching {domain} for metadata backfill...")
        products = refetch_products(domain)

        for product in products:
            catalogue_id = f"shopify_{domain}_{product['id']}"
            mask = df["catalogue_id"] == catalogue_id
            if not mask.any():
                continue
            if classify_category(product) is None:
                continue

            df.loc[mask, "subtype"] = extract_subtype(product.get("title", ""))
            df.loc[mask, "color_query"] = extract_color(product)
            updated += 1

    df.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)
    print(f"\nBackfilled {updated} rows.")
    print(df["subtype"].value_counts().head(10))
    print(df["color_query"].value_counts().head(10))


if __name__ == "__main__":
    main()