# Jewel Lens — Jewellery Catalogue Matcher

Upload a photo, get back the top-5 closest catalogue matches with confidence scores, shop, and price. Handles multiple jewellery items in one photo independently. See `DECISIONS.md` for architecture, trade-offs, and known limitations.

## Quickstart (< 5 minutes, clean machine)

This assumes the pre-built catalogue artifacts (`data/catalogue_metadata.parquet`, `data/catalogue_embeddings.npy`, `indices/catalogue.faiss`, `indices/id_mapping.pkl`, `models/category_classifier.pkl`) are already included in this project folder. If you're starting from raw scraped data instead, see **Full Rebuild** below — that takes 1–2 hours, not 5 minutes.

```bash
# 1. Clone / unzip the project, then from the project root:
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (first run downloads OWL-ViT + DINOv2 checkpoints, ~1-2GB — needs internet once)
python src/api/app.py
```

Open **http://localhost:5000** in your browser, upload a jewellery photo, and click **Find Matches**.

That's it. Everything else (model loading, catalogue indexing) is already done — the app just loads the pre-built files at startup.

### If it's slow on first run
The first `/predict` request after starting the server may take longer while models finish loading into memory. Subsequent requests are fast.

### Troubleshooting
- **Segmentation fault on macOS/Apple Silicon**: this project sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` at the top of every entry-point script to avoid a known PyTorch/FAISS OpenMP conflict. If you added a new entry-point script, add those two lines before any other imports.
- **`ModuleNotFoundError`**: make sure the virtual environment is activated (`source venv/bin/activate`) and you ran `pip install -r requirements.txt` inside it.
- **Port 5000 already in use**: change the port in `src/api/app.py`'s last line (`app.run(debug=True, port=5000)`) to something else, e.g. `5001`.

## Project structure

```
config.py                  # all paths, thresholds, model names in one place
requirements.txt

data/
  raw/                     # scraped catalogue images
  catalogue_metadata.parquet
  catalogue_embeddings.npy / _meta.parquet
  detect_test/, negatives/, stumper/   # test photos + ground-truth labels

indices/
  catalogue.faiss          # FAISS HNSW index
  id_mapping.pkl

models/
  category_classifier.pkl  # logistic regression on DINOv2 embeddings

src/
  scrape/                  # catalogue scraping (Shopify /products.json)
  catalogue/                # dedup, CLIP QA, embedding, index build, classifier training
  pipeline/                 # preprocess, detect, embed, classify, retrieve, decision, filter, run_pipeline
  api/                      # Flask app + response schema
  frontend/static/          # HTML/CSS/JS UI
  eval/                     # stumper eval harness + metrics (Phase 8)

DECISIONS.md               # architecture, trade-offs, known failure modes, testing results
```

## Full rebuild (from raw scrape — not needed for the 5-minute quickstart)

Only do this if you want to regenerate the catalogue from scratch (~1–2 hours, mostly DINOv2 embedding time on CPU):

```bash
# 1. Add your Etsy/Shopify store domains and API keys to config.py, then:
python src/scrape/shopify_scrapper.py        # scrape catalogue (~30-60 min depending on domain count)

# 2. Clean the catalogue
python src/catalogue/dedup_images.py          # perceptual-hash dedup
python src/catalogue/clip_label_qa.py         # CLIP zero-shot mismatch QA

# 3. Build embeddings + index
python src/catalogue/build_embeddings.py      # ~60-90 min on CPU for ~13k images
python src/catalogue/build_index.py

# 4. Train the category classifier
python src/catalogue/train_classifier.py

# 5. Then run the app as in Quickstart above
python src/api/app.py
```

## Running the evaluation (Phase 8)

```bash
python src/eval/harness.py
```

Reports Recall@1, Recall@5, MRR, Detection Recall, Category Accuracy, confused-pairs, and accuracy broken down by failure condition (lighting, angle, occlusion, multi-item, etc.), using the labelled stumper set in `data/stumper/`.

## Requirements

- Python 3.10+
- ~2GB free disk for model checkpoints (downloaded automatically on first run)
- No GPU required — everything runs on CPU
---

## v2 (2026-10): running, evaluating, reproducing

### Run the app

```bash
source venv2/bin/activate
PORT=5001 python src/api/app.py      # AirPlay often holds port 5000 on macOS; default is still 5000
```

Open http://localhost:5001. What's new in the UI and API:
- Each detected item shows its crop, the fused category with confidence, and a status with the reason (out-of-scope, low similarity).
- Results come in two tiers: **Closest matches** (score ≥ `HIGH_MATCH_THRESHOLD`) and **Similar items**.
- Every card has **More like this**.
- `POST /predict`: each item now also returns `crop_image`, `category_confidence`, `detector_category`, `status_reason`, `out_of_scope_probability`, `closest_matches` and `similar_items`. `results` is kept for compatibility.
- `GET /similar/<catalogue_id>?min_price=&max_price=&sort_by=relevance|price_low|price_high`: "more like this" for a catalogue listing (404 for unknown ids).
- `GET /catalogue-image/<path>` now only serves files under `data/raw/`.
- Prices are converted to INR at load time (`SHOP_CURRENCY`, `CURRENCY_TO_INR` in `config.py`).

### Configuration switches (`config.py`)

| setting | value | meaning |
|---|---|---|
| `CATEGORY_MODE` | `"fused"` | `"legacy"` restores v1 category and re-rank |
| `HIGH_MATCH_THRESHOLD` / `LOW_MATCH_THRESHOLD` / `MARGIN_THRESHOLD` | 0.74 / 0.475 / 0.0 | calibrated in Phase 8 (v1: 0.65 / 0.45 / 0.03) |
| `OOS_THRESHOLD` | 0.813 | CLIP out-of-scope check |
| `MMR_LAMBDA_PREDICT` / `MMR_LAMBDA_SIMILAR` | 1.0 / 0.7 | diversity re-rank off for photo search, on for "more like this" |

v2 artifacts: `artifacts/category_v2/` (retrained classifier, fusion weights, CLIP prompts) and `artifacts/openset/` (out-of-scope prompts, calibration). Experiments that were not adopted are kept under `artifacts/` (`dinov2_ft`, `dinov2_ftaug`, `dinov2_nobg`, `clip_b32`). Backups of every replaced file are in `artifacts_backup/<timestamp>/`.

### Evaluation (each script appends its tables to `results/LOG.md` and writes `results/<phase>.json`)

```bash
python src/eval/splits.py                      # fixed splits (already created; refuses to overwrite)
python src/eval/retrieval_eval.py --split val  # catalogue same-listing protocol + augmentations + pipeline mode
python src/eval/own_eval.py                    # own photos, holdout: counts, category, graded nDCG, negatives
python src/eval/category_fusion.py             # Phase 4 category signals and fusion (tune -> holdout)
python src/eval/detection_bench.py             # Phase 5 detectors / prompts / post-processing (needs detection caches)
python src/eval/embedding_eval.py              # Phase 6 embedding variants (needs artifacts/eval_cache)
python src/train/train_head.py                 # Phase 7 projection head (train split only)
python src/eval/calibrate.py                   # Phase 8 open-set thresholds
python src/eval/diversity.py                   # Phase 8 MMR lambda
python src/eval/final_eval.py                  # Phase 10: v1 vs v2 on holdout + test, latency per stage
```

Manual-labelling helpers: `python src/eval/prepare_own_eval.py` (label and grade pages), `--topup <detections.json> --name <x>` (grade only new results) and `--pool a.json,b.json --half tune --name <x>` (blind pooled grading).

Optional dependency for the background-removal experiment only: `pip install "rembg[cpu]"`. The production app does not need it.

Results summary: see `writeup.md` (Part C) and `results/LOG.md`.
