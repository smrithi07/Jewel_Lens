"""
Phase 1 — Etsy catalogue scraper.
Uses includes=Images to avoid per-listing API calls (stays well under the
5 QPS / 5000 QPD limit), with rate limiting, retry/backoff, and a resumable
checkpoint so a crash or quota stop doesn't lose progress.
"""

import os
import json
import time
import requests
import pandas as pd
import sys
sys.path.append(".")
import config

API_KEY = config.ETSY_API_KEY
BASE_URL = "https://openapi.etsy.com/v3/application/listings/active"
HEADERS = {"x-api-key": API_KEY}

_last_request_time = 0.0
_request_count = 0


def _rate_limited_get(url, params):
    """Enforces REQUESTS_PER_SECOND, retries on 429/5xx with backoff, tracks daily budget."""
    global _last_request_time, _request_count

    if _request_count >= config.DAILY_REQUEST_BUDGET - config.REQUEST_BUDGET_BUFFER:
        raise RuntimeError("DAILY_BUDGET_REACHED")

    min_interval = 1.0 / config.REQUESTS_PER_SECOND
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params)
        _last_request_time = time.time()
        _request_count += 1

        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"    [429] rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        if 500 <= resp.status_code < 600:
            wait = 2 ** attempt
            print(f"    [{resp.status_code}] server error, retrying in {wait}s")
            time.sleep(wait)
            continue
        print(f"    [warn] HTTP {resp.status_code}: {resp.text[:150]}")
        return resp
    return resp


def search_listings(query, max_results):
    """Page through search results with includes=Images — no per-listing call needed."""
    results = []
    offset = 0
    while len(results) < max_results:
        params = {
            "keywords": query,
            "limit": min(100, max_results - len(results)),
            "offset": offset,
            "includes": "Images",
        }
        resp = _rate_limited_get(BASE_URL, params)
        if resp.status_code != 200:
            break
        batch = resp.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        offset += len(batch)
    return results[:max_results]


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


def load_progress():
    if os.path.exists(config.SCRAPE_PROGRESS_PATH):
        with open(config.SCRAPE_PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def save_progress(done_facets):
    with open(config.SCRAPE_PROGRESS_PATH, "w") as f:
        json.dump(list(done_facets), f)


def append_partial(rows):
    with open(config.SCRAPE_PARTIAL_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def scrape_facet(category, facet_type, facet_value, max_results):
    query = f"{facet_value} {category}"
    print(f"Searching: {query!r} ({facet_type}, target {max_results})")
    listings = search_listings(query, max_results)

    rows = []
    for listing in listings:
        listing_id = listing["listing_id"]
        title = listing.get("title", "")
        price_info = listing.get("price", {})
        price = price_info.get("amount", 0) / max(price_info.get("divisor", 1), 1)
        url = listing.get("url", "")
        images = listing.get("images", [])  # embedded via includes=Images, no extra call
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
    print(f"  -> got {len(rows)} listings, {sum(len(r['image_paths'].split(';')) for r in rows)} images "
          f"[{_request_count} API calls used]")
    return rows


def main():
    os.makedirs(config.RAW_IMAGE_DIR, exist_ok=True)
    done_facets = load_progress()

    facets_to_run = []
    for category, facet in config.SEARCH_FACETS.items():
        for subtype in facet.get("subtypes", []):
            facets_to_run.append((category, "subtype", subtype, config.TARGET_PER_SUBTYPE))
        for color in facet.get("colors", []):
            facets_to_run.append((category, "color", color, config.TARGET_PER_COLOR))

    for category, facet_type, value, target in facets_to_run:
        key = f"{category}|{facet_type}|{value}"
        if key in done_facets:
            continue
        try:
            rows = scrape_facet(category, facet_type, value, target)
            append_partial(rows)
            done_facets.add(key)
            save_progress(done_facets)
        except RuntimeError as e:
            if str(e) == "DAILY_BUDGET_REACHED":
                print(f"\n[stop] Daily request budget reached ({_request_count} calls used). "
                      f"Progress saved — rerun this script tomorrow to resume where it left off.")
                break
            raise

    # Consolidate partial results into the final parquet
    if os.path.exists(config.SCRAPE_PARTIAL_PATH):
        rows = [json.loads(line) for line in open(config.SCRAPE_PARTIAL_PATH)]
        df = pd.DataFrame(rows).drop_duplicates(subset="catalogue_id")
        df.to_parquet(config.CATALOGUE_METADATA_PATH, index=False)
        print(f"\nTotal listings so far: {len(df)}")
        print(f"Total images: {df['image_paths'].apply(lambda p: len(p.split(';'))).sum()}")
        print(f"Facets completed: {len(done_facets)} / {len(facets_to_run)}")
        print("\nCategory distribution:")
        print(df["category"].value_counts())


if __name__ == "__main__":
    main()