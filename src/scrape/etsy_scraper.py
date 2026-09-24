"""
Phase 1 — Etsy catalogue scraper.
Scrapes by category x subtype x color facet, downloads ALL images per
listing (tagged to the same catalogue_id) for real same-item photo pairs,
and writes a metadata table.
"""

import os
import time
import requests
import pandas as pd
import sys
sys.path.append(".")
import config

API_KEY = config.ETSY_API_KEY
BASE_URL = "https://openapi.etsy.com/v3/application/listings/active"
HEADERS = {"x-api-key": API_KEY}


def search_listings(query, limit=100, max_results=120):
    results = []
    offset = 0
    while len(results) < max_results:
        params = {"keywords": query, "limit": min(limit, max_results - len(results)), "offset": offset}
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"  [warn] {query!r} -> HTTP {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        offset += len(batch)
        time.sleep(0.2)
    return results[:max_results]


def get_listing_images(listing_id):
    url = f"https://openapi.etsy.com/v3/application/listings/{listing_id}/images"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return []
    return resp.json().get("results", [])


def download_image(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def scrape_facet(category, facet_type, facet_value, max_results):
    query = f"{facet_value} {category}"
    print(f"Searching: {query!r} ({facet_type}, target {max_results})")
    listings = search_listings(query, max_results=max_results)

    rows = []
    for listing in listings:
        listing_id = listing["listing_id"]
        title = listing.get("title", "")
        price_info = listing.get("price", {})
        price = price_info.get("amount", 0) / max(price_info.get("divisor", 1), 1)
        url = listing.get("url", "")

        images = get_listing_images(listing_id)
        if not images:
            continue

        item_dir = os.path.join(config.RAW_IMAGE_DIR, str(listing_id))
        os.makedirs(item_dir, exist_ok=True)

        saved_images = []
        for img in images:
            img_url = img.get("url_570xN") or img.get("url_fullxfull")
            if not img_url:
                continue
            img_path = os.path.join(item_dir, f"{img['listing_image_id']}.jpg")
            if download_image(img_url, img_path):
                saved_images.append(img_path)

        if not saved_images:
            continue

        rows.append({
            "catalogue_id": listing_id,
            "category": category,
            "subtype": facet_value if facet_type == "subtype" else "",
            "color_query": facet_value if facet_type == "color" else "",
            "title": title,
            "price": price,
            "listing_url": url,
            "image_paths": ";".join(saved_images),
            "primary_image": saved_images[0],
        })
    print(f"  -> got {len(rows)} listings, {sum(len(r['image_paths'].split(';')) for r in rows)} images")
    return rows


def main():
    os.makedirs(config.RAW_IMAGE_DIR, exist_ok=True)
    all_rows = []

    for category, facet in config.SEARCH_FACETS.items():
        for subtype in facet.get("subtypes", []):
            all_rows.extend(scrape_facet(category, "subtype", subtype, config.TARGET_PER_SUBTYPE))
        for color in facet.get("colors", []):
            all_rows.extend(scrape_facet(category, "color", color, config.TARGET_PER_COLOR))

    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset="catalogue_id", inplace=True)
    df.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)

    print(f"\nTotal listings: {len(df)}")
    print(f"Total images: {df['image_paths'].apply(lambda p: len(p.split(';'))).sum()}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nSubtype distribution:")
    print(df["subtype"].value_counts())
    print("\nColor-query distribution:")
    print(df["color_query"].value_counts())


if __name__ == "__main__":
    main()