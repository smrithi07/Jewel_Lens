"""
Phase 1 (backup) — Shopify /products.json scraper.
No auth needed. Filters to your 3 jewelry categories via strict,
word-boundary-matched keywords, with an exclude check for watches.
Uses collection-based scraping where STORE_COLLECTIONS has a confirmed
handle, else falls back to keyword-filtered whole-catalogue scanning.
Writes to parquet incrementally (per domain) so an interrupt never
loses more than the store currently in progress. Skips re-downloading
images already on disk, so reruns are cheap.
"""

import os
import re
import time
import requests
import pandas as pd
import sys
sys.path.append(".")
import config
import json

_last_request_time = 0.0
REQUESTS_PER_SECOND = 2


def _rate_limited_get(url, params=None):
    global _last_request_time
    min_interval = 1.0 / REQUESTS_PER_SECOND
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"    [warn] network error: {e}")
        return None
    _last_request_time = time.time()
    return resp

def load_completed_domains():
    if os.path.exists(config.SHOPIFY_DOMAIN_PROGRESS_PATH):
        with open(config.SHOPIFY_DOMAIN_PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def save_completed_domains(completed):
    with open(config.SHOPIFY_DOMAIN_PROGRESS_PATH, "w") as f:
        json.dump(list(completed), f)


def classify_category(product):
    """Strict word-boundary match against CATEGORY_KEYWORDS, with an
    exclude check first so e.g. 'watch bracelet stack' never matches."""
    text = " ".join([
        product.get("product_type", ""),
        product.get("title", ""),
        " ".join(product.get("tags", [])) if isinstance(product.get("tags"), list) else str(product.get("tags", "")),
    ]).lower()

    if any(kw in text for kw in config.EXCLUDE_KEYWORDS):
        return None

    for category, keywords in config.CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return category
    return None


def extract_color(product):
    options = product.get("options", [])
    color_idx = None
    for i, opt in enumerate(options):
        if opt.get("name", "").lower() in ("color", "colour"):
            color_idx = i
            break
    if color_idx is None:
        return ""
    variant = product.get("variants", [{}])[0]
    return variant.get(f"option{color_idx + 1}", "") or ""


def extract_subtype(title):
    title_lower = title.lower()
    for subtype, keywords in config.SUBTYPE_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return subtype
    return ""


def download_image(url, path):
    if os.path.exists(path):
        return True  # already downloaded, skip
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def _build_row(domain, product, category):
    images = product.get("images", [])
    if not images:
        return None

    product_id = product["id"]
    item_dir = os.path.join(config.RAW_IMAGE_DIR, f"shopify_{domain}_{product_id}")
    os.makedirs(item_dir, exist_ok=True)

    saved_images = []
    for img in images:
        img_url = img.get("src")
        if not img_url:
            continue
        img_path = os.path.join(item_dir, f"{img['id']}.jpg")
        if download_image(img_url, img_path):
            saved_images.append(img_path)
    if not saved_images:
        return None

    variant = product.get("variants", [{}])[0]
    return {
        "catalogue_id": f"shopify_{domain}_{product_id}",
        "category": category,
        "subtype": extract_subtype(product.get("title", "")),
        "color_query": extract_color(product),
        "title": product.get("title", ""),
        "price": float(variant.get("price", 0)),
        "listing_url": f"https://{domain}/products/{product.get('handle', '')}",
        "image_paths": ";".join(saved_images),
        "primary_image": saved_images[0],
    }


def scrape_collection(domain, category, handle, max_products):
    print(f"  Collection: {handle} -> {category}")
    rows = []
    page = 1
    while len(rows) < max_products:
        url = f"https://{domain}/collections/{handle}/products.json"
        resp = _rate_limited_get(url, params={"limit": 250, "page": page})
        if resp is None or resp.status_code != 200:
            break
        products = resp.json().get("products", [])
        if not products:
            break
        for product in products:
            if len(rows) >= max_products:
                break
            text = " ".join([product.get("title", ""), product.get("product_type", "")]).lower()
            if any(kw in text for kw in config.EXCLUDE_KEYWORDS):
                continue
            row = _build_row(domain, product, category)
            if row:
                rows.append(row)
        page += 1
    print(f"    -> {len(rows)} items")
    return rows


def scrape_whole_catalogue(domain, max_pages, max_products):
    rows = []
    page = 1
    while page <= max_pages and len(rows) < max_products:
        url = f"https://{domain}/products.json"
        resp = _rate_limited_get(url, params={"limit": 250, "page": page})
        if resp is None or resp.status_code != 200:
            print(f"  [warn] page {page}: HTTP {resp.status_code}")
            break
        products = resp.json().get("products", [])
        if not products:
            break

        for product in products:
            if len(rows) >= max_products:
                break
            category = classify_category(product)
            if category is None:
                continue
            row = _build_row(domain, product, category)
            if row:
                rows.append(row)

        print(f"  page {page}: {len(products)} products scanned, {len(rows)} jewelry matches so far")
        page += 1

    return rows


def scrape_store(domain, max_pages=50):
    print(f"\n=== Scraping {domain} ===")
    if domain in config.STORE_COLLECTIONS:
        rows = []
        collections = config.STORE_COLLECTIONS[domain]
        per_category_cap = config.MAX_PRODUCTS_PER_STORE // len(collections)
        for category, handle in collections.items():
            rows.extend(scrape_collection(domain, category, handle, per_category_cap))
        return rows
    return scrape_whole_catalogue(domain, max_pages, config.MAX_PRODUCTS_PER_STORE)

def domain_has_products_json(domain):
    try:
        resp = requests.get(f"https://{domain}/products.json", params={"limit": 1}, timeout=10)
        return resp.status_code == 200 and "products" in resp.json()
    except (requests.RequestException, ValueError):
        return False


def main():
    os.makedirs(config.RAW_IMAGE_DIR, exist_ok=True)
    completed = load_completed_domains()

    for domain in config.SHOPIFY_STORE_DOMAINS:
        if domain in completed:
            print(f"\n=== Skipping {domain} — already completed ===")
            continue

        if not domain_has_products_json(domain):
            print(f"\n=== Skipping {domain} — /products.json not available ===")
            completed.add(domain)
            save_completed_domains(completed)
            continue

        rows = scrape_store(domain)
        df_new = pd.DataFrame(rows).drop_duplicates(subset="catalogue_id")

        if os.path.exists(config.CATALOGUE_METADATA_PATH):
            existing = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
            df_new = pd.concat([existing, df_new]).drop_duplicates(subset="catalogue_id")

        df_new.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)
        print(f"  [saved] {domain} -> {len(df_new)} total rows in catalogue so far")

        completed.add(domain)
        save_completed_domains(completed)

    df = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    print(f"\nTotal listings: {len(df)}")
    print(f"Total images: {df['image_paths'].apply(lambda p: len(p.split(';'))).sum()}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nSubtype distribution:")
    print(df["subtype"].value_counts())


if __name__ == "__main__":
    main()