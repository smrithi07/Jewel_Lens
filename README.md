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