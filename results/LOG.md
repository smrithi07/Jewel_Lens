# Jewel Lens v2 — results log

One section per phase. Machine-readable numbers live in `results/<phase>.json`.
Splits: `data/splits.json` (catalogue, listing-level) and `data/stumper/split.json` (stumper tune/holdout), both created in Phase 1.

## Phase 0 — Recon and safety (2026-10-10)

No change to behaviour.

| Item | Value |
|---|---|
| Backup | `artifacts_backup/20261010_192617/` (84 MB, with SHA256SUMS) |
| Backed-up files | `indices/catalogue.faiss`, `indices/id_mapping.pkl`, `models/category_classifier.pkl`, `data/catalogue_embeddings.npy`, `data/catalogue_embeddings_meta.parquet`, `data/catalogue_metadata.parquet`, `config.py` |
| Listings / indexed images | 3,513 / 13,352 (earring 6,400, necklace 3,760, bracelet 3,192 images) |
| Listings with ≥2 images (eligible queries) | 3,277 (earring 1,633, necklace 888, bracelet 756) |
| Stumper photos | 136 (earring 67, necklace 22, bracelet 23, mixed 13, negatives 11), plus 6 in `data/negatives/` |
| API smoke test | `python src/api/app.py`: `GET /` gives 200, `POST /predict` gives 200 with an `items` list |

Baseline figures carried over from v1 (not re-measured): stumper Detection Recall 1.000, Category Accuracy 0.665, No-match precision 0.583.

## Phase 1 — baseline, stumper photos (2026-10-10T19:51:53)

Ground truth = folder name. `mixed` counts toward detection recall only. 95% bootstrap CIs by photo. **Headline = holdout.** `as_deployed` = what the API does today (no EXIF/resize); `with_preprocess` = preprocess_image() first.

| input | half | photos | Det. recall | Category acc (v1 def.) | Detector-label acc | Item category acc | Neg. rejected | Pos. false-reject |
|---|---|---|---|---|---|---|---|---|
| as_deployed | holdout | 73 | 0.984 [0.953, 1.000] | 0.589 [0.464, 0.732] | 0.857 [0.750, 0.929] | 0.589 [0.464, 0.732] | 0.444 [0.111, 0.778] (n=9) | 0.000 [0.000, 0.000] |
| as_deployed | tune | 69 | 0.934 [0.869, 0.984] | 0.588 [0.451, 0.725] | 0.784 [0.667, 0.902] | 0.588 [0.451, 0.725] | 0.875 [0.625, 1.000] (n=8) | 0.016 [0.000, 0.049] |
| as_deployed | all | 142 | 0.960 [0.920, 0.992] | 0.589 [0.495, 0.682] | 0.822 [0.748, 0.897] | 0.589 [0.495, 0.682] | 0.647 [0.412, 0.882] (n=17) | 0.008 [0.000, 0.024] |
| with_preprocess | holdout | 73 | 0.984 [0.953, 1.000] | 0.589 [0.464, 0.732] | 0.857 [0.750, 0.929] | 0.589 [0.464, 0.732] | 0.444 [0.111, 0.778] (n=9) | 0.000 [0.000, 0.000] |
| with_preprocess | tune | 69 | 0.934 [0.869, 0.984] | 0.588 [0.451, 0.725] | 0.784 [0.667, 0.902] | 0.588 [0.451, 0.725] | 0.875 [0.625, 1.000] (n=8) | 0.016 [0.000, 0.049] |
| with_preprocess | all | 142 | 0.960 [0.920, 0.992] | 0.589 [0.495, 0.682] | 0.822 [0.748, 0.897] | 0.589 [0.495, 0.682] | 0.647 [0.412, 0.882] (n=17) | 0.008 [0.000, 0.024] |

Holdout confusion, as_deployed (rows = true folder, first item):

| true | classifier → earring / necklace / bracelet | detector → earring / necklace / bracelet |
|---|---|---|
| earring | 33 / 0 / 0 | 30 / 2 / 1 |
| necklace | 11 / 0 / 0 | 0 / 11 / 0 |
| bracelet | 12 / 0 / 0 | 4 / 1 / 7 |

Pipeline latency per photo (ms): as_deployed p50 409 / p95 639, with_preprocess p50 407 / p95 628

## Phase 1 — baseline, split = val (2026-10-10T19:53:04)

Same-listing protocol: query image excluded from the index. Variant `prod` is what is deployed now. `fixed_input` is a diagnostic (classifier fed the un-normalised embedding it was trained on), not a tuned change. Graded nDCG@5 is **provisional** until Phase 2 backfills subtype.

**Retrieval-only, clean, all eligible queries**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (prov.) | cat P@5 (non-set) | R@1 pHash | R@5 pHash |
|---|---|---|---|---|---|---|---|---|---|---|
| raw (all) | 502 | 0.297 | 0.424 | 0.472 | 0.351 | 0.362 | 0.704 | 0.959 | 0.311 | 0.462 |
| prod (all) | 502 | 0.293 | 0.424 | 0.472 | 0.349 | 0.360 | 0.694 | 0.953 | 0.307 | 0.462 |
| fixed_input (all) | 502 | 0.295 | 0.428 | 0.474 | 0.351 | 0.363 | 0.715 | 0.979 | 0.309 | 0.466 |
| raw (non-set) | 455 | 0.297 | 0.420 | 0.470 | 0.350 | 0.360 | 0.703 | 0.959 | 0.310 | 0.459 |
| prod (non-set) | 455 | 0.295 | 0.420 | 0.470 | 0.349 | 0.359 | 0.699 | 0.953 | 0.308 | 0.459 |
| fixed_input (non-set) | 455 | 0.295 | 0.424 | 0.470 | 0.350 | 0.361 | 0.714 | 0.979 | 0.308 | 0.464 |

prod 95% bootstrap CI: R@1 0.293 [0.251, 0.331], R@5 0.424 [0.382, 0.466], MRR 0.349 [0.309, 0.384]
Queries whose image has a cross-listing pHash near-duplicate: 0.143
Classifier accuracy on the query image (non-set): prod input 0.534, fixed input 0.980

**Per category (prod, non-set)**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (prov.) | cat P@5 (non-set) | R@1 pHash | R@5 pHash |
|---|---|---|---|---|---|---|---|---|---|---|
| earring | 243 | 0.288 | 0.436 | 0.461 | 0.343 | 0.362 | 0.711 | 0.989 | 0.296 | 0.465 |
| necklace | 108 | 0.213 | 0.315 | 0.417 | 0.274 | 0.270 | 0.638 | 0.898 | 0.222 | 0.352 |
| bracelet | 104 | 0.394 | 0.490 | 0.548 | 0.442 | 0.444 | 0.736 | 0.923 | 0.423 | 0.558 |

**Per augmentation (retrieval-only, re-embedded, fixed 300-query subset)**

| augmentation | R@1 raw | R@1 prod | R@1 fixed | R@5 prod | MRR prod | cat P@5 prod |
|---|---|---|---|---|---|---|
| clean | 0.313 | 0.307 | 0.310 | 0.450 | 0.365 | 0.951 |
| blur | 0.137 | 0.137 | 0.137 | 0.230 | 0.179 | 0.822 |
| low_light | 0.283 | 0.280 | 0.280 | 0.420 | 0.338 | 0.948 |
| bright_contrast | 0.293 | 0.287 | 0.287 | 0.410 | 0.343 | 0.942 |
| rotate | 0.260 | 0.253 | 0.257 | 0.377 | 0.315 | 0.941 |
| crop_zoom | 0.277 | 0.273 | 0.277 | 0.400 | 0.329 | 0.929 |
| jpeg | 0.323 | 0.317 | 0.320 | 0.433 | 0.367 | 0.949 |
| busy_bg | 0.077 | 0.080 | 0.080 | 0.143 | 0.115 | 0.817 |
| perspective | 0.280 | 0.270 | 0.277 | 0.417 | 0.339 | 0.949 |
| combined | 0.140 | 0.140 | 0.143 | 0.210 | 0.171 | 0.804 |

**Full pipeline (detect → crop → embed → retrieve), 300-query subset, scored on top detection**

| input | det. rate | items/img | detector label acc | R@1 prod | R@5 prod | MRR prod | R@1 fixed | R@5 best-of-items (prod) |
|---|---|---|---|---|---|---|---|---|
| clean | 0.977 | 1.85 | 0.706 | 0.283 | 0.427 | 0.341 | 0.277 | 0.480 |
| combined | 0.623 | 1.16 | 0.684 | 0.133 | 0.217 | 0.169 | 0.140 | 0.240 |

**Latency (ms)**

| stage | p50 | p95 | n |
|---|---|---|---|
| search | 0.1 | 0.2 | 502 |
| classify_rerank | 0.1 | 0.1 | 502 |
| embed_amortised_batch16 | 81.6 | 105.8 | 190 |
| pipeline_preprocess | 1.4 | 51.5 | 600 |
| pipeline_detect | 315.1 | 348.5 | 600 |
| pipeline_crop | 0.3 | 1.3 | 905 |
| pipeline_embed | 122.7 | 166.9 | 905 |
| pipeline_search | 0.2 | 0.2 | 905 |
| pipeline_classify_rerank | 0.2 | 0.2 | 905 |
| pipeline_decide | 0.0 | 0.0 | 905 |

## Phase 1 — baseline, split = test (2026-10-10T20:08:17)

Same-listing protocol: query image excluded from the index. Variant `prod` is what is deployed now. `fixed_input` is a diagnostic (classifier fed the un-normalised embedding it was trained on), not a tuned change. Graded nDCG@5 is **provisional** until Phase 2 backfills subtype.

**Retrieval-only, clean, all eligible queries**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (prov.) | cat P@5 (non-set) | R@1 pHash | R@5 pHash |
|---|---|---|---|---|---|---|---|---|---|---|
| raw (all) | 493 | 0.316 | 0.440 | 0.497 | 0.378 | 0.385 | 0.720 | 0.962 | 0.329 | 0.471 |
| prod (all) | 493 | 0.314 | 0.438 | 0.497 | 0.376 | 0.383 | 0.713 | 0.957 | 0.327 | 0.469 |
| fixed_input (all) | 493 | 0.318 | 0.440 | 0.497 | 0.379 | 0.386 | 0.731 | 0.980 | 0.331 | 0.471 |
| raw (non-set) | 440 | 0.323 | 0.436 | 0.493 | 0.382 | 0.387 | 0.723 | 0.962 | 0.334 | 0.466 |
| prod (non-set) | 440 | 0.320 | 0.436 | 0.493 | 0.380 | 0.386 | 0.718 | 0.957 | 0.332 | 0.466 |
| fixed_input (non-set) | 440 | 0.323 | 0.436 | 0.493 | 0.382 | 0.387 | 0.733 | 0.980 | 0.334 | 0.466 |

prod 95% bootstrap CI: R@1 0.314 [0.274, 0.355], R@5 0.438 [0.394, 0.481], MRR 0.376 [0.337, 0.414]
Queries whose image has a cross-listing pHash near-duplicate: 0.128
Classifier accuracy on the query image (non-set): prod input 0.523, fixed input 0.989

**Per category (prod, non-set)**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (prov.) | cat P@5 (non-set) | R@1 pHash | R@5 pHash |
|---|---|---|---|---|---|---|---|---|---|---|
| earring | 230 | 0.361 | 0.478 | 0.517 | 0.416 | 0.427 | 0.741 | 0.986 | 0.374 | 0.500 |
| necklace | 105 | 0.248 | 0.314 | 0.362 | 0.293 | 0.288 | 0.669 | 0.928 | 0.267 | 0.371 |
| bracelet | 105 | 0.305 | 0.467 | 0.571 | 0.390 | 0.394 | 0.716 | 0.924 | 0.305 | 0.486 |

**Per augmentation (retrieval-only, re-embedded, fixed 300-query subset)**

| augmentation | R@1 raw | R@1 prod | R@1 fixed | R@5 prod | MRR prod | cat P@5 prod |
|---|---|---|---|---|---|---|
| clean | 0.320 | 0.320 | 0.320 | 0.427 | 0.376 | 0.956 |
| blur | 0.133 | 0.133 | 0.133 | 0.233 | 0.183 | 0.856 |
| low_light | 0.300 | 0.300 | 0.300 | 0.420 | 0.355 | 0.948 |
| bright_contrast | 0.280 | 0.280 | 0.280 | 0.403 | 0.343 | 0.949 |
| rotate | 0.277 | 0.280 | 0.280 | 0.410 | 0.341 | 0.943 |
| crop_zoom | 0.280 | 0.277 | 0.280 | 0.423 | 0.341 | 0.937 |
| jpeg | 0.313 | 0.313 | 0.313 | 0.427 | 0.365 | 0.946 |
| busy_bg | 0.083 | 0.083 | 0.087 | 0.183 | 0.129 | 0.805 |
| perspective | 0.303 | 0.303 | 0.303 | 0.427 | 0.360 | 0.952 |
| combined | 0.140 | 0.137 | 0.130 | 0.230 | 0.186 | 0.833 |

**Full pipeline (detect → crop → embed → retrieve), 300-query subset, scored on top detection**

| input | det. rate | items/img | detector label acc | R@1 prod | R@5 prod | MRR prod | R@1 fixed | R@5 best-of-items (prod) |
|---|---|---|---|---|---|---|---|---|
| clean | 0.970 | 1.81 | 0.739 | 0.300 | 0.410 | 0.353 | 0.287 | 0.463 |
| combined | 0.610 | 1.15 | 0.710 | 0.140 | 0.227 | 0.175 | 0.133 | 0.247 |

**Latency (ms)**

| stage | p50 | p95 | n |
|---|---|---|---|
| search | 0.1 | 0.2 | 493 |
| classify_rerank | 0.1 | 0.1 | 493 |
| embed_amortised_batch16 | 82.1 | 112.7 | 190 |
| pipeline_preprocess | 1.4 | 58.7 | 600 |
| pipeline_detect | 319.4 | 353.3 | 600 |
| pipeline_crop | 0.3 | 1.3 | 886 |
| pipeline_embed | 112.8 | 161.8 | 886 |
| pipeline_search | 0.2 | 0.2 | 886 |
| pipeline_classify_rerank | 0.2 | 0.2 | 886 |
| pipeline_decide | 0.0 | 0.0 | 886 |

## Phase 1 — summary (side by side)

| Benchmark | Metric | Baseline (prod, as deployed) |
|---|---|---|
| Catalogue retrieval, val (n=502) | R@1 / R@5 / R@10 / MRR / nDCG@5 | 0.293 [0.251, 0.331] / 0.424 / 0.472 / 0.349 / 0.360 |
| Catalogue retrieval, test (n=493, run once) | R@1 / R@5 / R@10 / MRR / nDCG@5 | 0.314 [0.274, 0.355] / 0.438 / 0.497 / 0.376 / 0.383 |
| Catalogue, val, combined phone-photo aug (n=300) | R@1 / R@5 | 0.140 / 0.210 |
| Catalogue, val, full pipeline clean / combined (n=300) | R@5 | 0.427 / 0.217 (detection rate 0.977 / 0.623) |
| Catalogue, val | category P@5 (non-set) | 0.953 |
| Stumper holdout (73 photos) | Detection recall | 0.984 [0.953, 1.000] |
| Stumper holdout | Category acc (v1 def.) | 0.589 [0.464, 0.732] (v1 reported 0.665 on all labelled rows) |
| Stumper holdout | Detector-label acc (what the UI shows) | 0.857 [0.750, 0.929] |
| Stumper holdout | Negatives rejected | 0.444 [0.111, 0.778] (n=9) |

Findings:
1. **Classifier input bug (root cause of failure mode 1).** `train_classifier.py` fits on raw DINOv2 CLS vectors (norm ≈ 46). `classify()` receives L2-normalised vectors (norm 1), so `w·x` shrinks about 46× and the intercepts decide. Live, it predicts "earring" for all 13,352 catalogue images and for every stumper item. Fed the input it was trained on, it scores 0.980 on val query images (non-set). The live re-rank therefore always boosts earring listings by about +0.064. The effect on catalogue retrieval is small (R@1 0.293 prod vs 0.297 raw), because top-5 results are already 95% the same category. Its main damage is category display and stumper category accuracy.
2. Stumper category accuracy is close to the share of earrings in the set, because the classifier is constant. The OWL-ViT label (0.857) beats it.
3. The API (`src/api/app.py`) never calls `preprocess_image()`. All stumper photos are already upright and ≤1600 px, so this set cannot measure the effect, but raw phone uploads with EXIF rotation will be sideways.
4. Same-listing retrieval is hard because listings mix model/lifestyle shots with product shots (see the worst-20 report). Background is the biggest robustness gap (busy_bg R@1 0.080), then blur (0.137).
5. 14% of val query images have a pHash near-duplicate in a different listing (cross-shop re-uploads). Counting those as relevant raises R@5 from 0.424 to 0.462.
6. The test split was evaluated once. Do not rerun it under the `baseline` tag (the script refuses).


## Phase 2 — metadata backfill (2026-10-10T21:03:47)

Fills blanks only, into `data/catalogue_metadata_backfilled.parquet` (production parquet untouched). Model threshold calibrated on train-split listings with known labels, target precision 0.85; reported on val-split known listings. Spot-check result pending (🙋).

| attribute | blank before | blank after | filled: variant / title / model | model | threshold | model val precision @t | model val coverage @t |
|---|---|---|---|---|---|---|---|
| subtype | 2117 | 620 | 0 / 529 / 968 | probe_clip+dino_mean | 0.42 (calibrated) | 0.857 | 0.785 |
| color | 3041 | 495 | 465 / 2344 / 209 | probe_clip+dino_mean | 0.42 (calibrated) | 0.911 | 0.527 |

Classes never filled by the model (train precision < min or < 10 samples): subtype: beaded, charm, huggie, layered, link, statement; color: black, multicolour, oxidized, pink, platinum, purple, red, white, yellow

Variant colour strings → palette:

| raw | palette |
|---|---|
| 14K Gold Waterproof | gold |
| 18K Gold Waterproof | gold |
| Black | black |
| Blue | blue |
| Blue::Gold | blue |
| Clear | (blank → title/model) |
| Gold | gold |
| Gold & White | white |
| Gold Stainless Steel WATERPROOF | gold |
| Gold Waterproof Stainless Steel | gold |
| Gold::Silver | (blank → title/model) |
| Green | green |
| Maroon | red |
| Multi color | multicolour |
| Multicolor | multicolour |
| Navy | blue |
| Opal | (blank → title/model) |
| Pink | pink |
| Rose Gold | rose gold |
| Ruby | red |
| SIlver | silver |
| Silver | silver |
| Silver Plated | silver |
| Solid | (blank → title/model) |
| Water Resistant - 14K Real Gold Plating | gold |
| Water Resistant - Gold / Silver | (blank → title/model) |
| Waterproof - 14K Real Gold Plating | gold |
| Waterproof - 18K Gold Waterproof | gold |
| Waterproof - 18K Real Gold Plating | gold |
| White | white |
| multicolor | multicolour |
| silver | silver |

## Phase 2 — baseline_bf, split = val (2026-10-10T21:03:04)

Same-listing protocol: query image excluded from the index. Variant `prod` is what is deployed now. `fixed_input` is a diagnostic (classifier fed the un-normalised embedding it was trained on), not a tuned change. Metadata: `data/catalogue_metadata_backfilled.parquet` — graded nDCG@5: subtype from data/catalogue_metadata_backfilled.parquet. Colour/subtype P@5 are over queries whose own attribute is known; blank results count as non-matching. *(trusted)* columns use only original/variant/title labels (no image-model fills) — use these to compare retrievers.

**Retrieval-only, clean, all eligible queries**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 | cat P@5 (non-set) | R@1 pHash | R@5 pHash | colour P@5 | subtype P@5 | graded nDCG@5 (trusted) | colour P@5 (trusted) | subtype P@5 (trusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw (all) | 502 | 0.297 | 0.424 | 0.472 | 0.351 | 0.362 | 0.740 | 0.959 | 0.311 | 0.462 | 0.490 | 0.630 | 0.710 | 0.470 | 0.522 |
| prod (all) | 502 | 0.293 | 0.424 | 0.472 | 0.349 | 0.360 | 0.730 | 0.953 | 0.307 | 0.462 | 0.488 | 0.622 | 0.701 | 0.468 | 0.517 |
| fixed_input (all) | 502 | 0.295 | 0.428 | 0.474 | 0.351 | 0.363 | 0.752 | 0.979 | 0.309 | 0.466 | 0.488 | 0.639 | 0.721 | 0.468 | 0.528 |
| raw (non-set) | 455 | 0.297 | 0.420 | 0.470 | 0.350 | 0.360 | 0.740 | 0.959 | 0.310 | 0.459 | 0.491 | 0.630 | 0.709 | 0.471 | 0.519 |
| prod (non-set) | 455 | 0.295 | 0.420 | 0.470 | 0.349 | 0.359 | 0.735 | 0.953 | 0.308 | 0.459 | 0.488 | 0.623 | 0.705 | 0.469 | 0.515 |
| fixed_input (non-set) | 455 | 0.295 | 0.424 | 0.470 | 0.350 | 0.361 | 0.751 | 0.979 | 0.308 | 0.464 | 0.489 | 0.639 | 0.719 | 0.469 | 0.524 |

prod 95% bootstrap CI: R@1 0.293 [0.251, 0.331], R@5 0.424 [0.382, 0.466], MRR 0.349 [0.309, 0.384]
Queries whose image has a cross-listing pHash near-duplicate: 0.143
Classifier accuracy on the query image (non-set): prod input 0.534, fixed input 0.980

**Per category (prod, non-set)**

| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 | cat P@5 (non-set) | R@1 pHash | R@5 pHash | colour P@5 | subtype P@5 | graded nDCG@5 (trusted) | colour P@5 (trusted) | subtype P@5 (trusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| earring | 243 | 0.288 | 0.436 | 0.461 | 0.343 | 0.362 | 0.759 | 0.989 | 0.296 | 0.465 | 0.459 | 0.651 | 0.721 | 0.438 | 0.540 |
| necklace | 108 | 0.213 | 0.315 | 0.417 | 0.274 | 0.270 | 0.686 | 0.898 | 0.222 | 0.352 | 0.534 | 0.578 | 0.656 | 0.506 | 0.424 |
| bracelet | 104 | 0.394 | 0.490 | 0.548 | 0.442 | 0.444 | 0.729 | 0.923 | 0.423 | 0.558 | 0.497 | 0.595 | 0.720 | 0.490 | 0.525 |

**Latency (ms)**

| stage | p50 | p95 | n |
|---|---|---|---|
| search | 0.1 | 0.2 | 502 |
| classify_rerank | 0.1 | 0.1 | 502 |

## Phase 2 — spot-check and final graded baseline (2026-10-10)

🙋 User spot-check of 60 backfilled listings (stratified by attribute × source):

| attribute | title | probe | total |
|---|---|---|---|
| subtype | 16/17 | 26/27 | 42/44 (95%) |
| colour | 35/36 | 18/19 | 53/55 (96%) |

All sources clear the 80% bar, so the calibrated thresholds stay. All four errors are edge cases: a variant-colour list in a title, a hand chain and a slider bracelet that the subtype list has no slot for, and one catalogue category error on a set listing.

**Final graded baseline (val, prod, backfilled metadata):** graded nDCG@5 0.730 (trusted labels: 0.701), colour P@5 0.488 (trusted 0.468), subtype P@5 0.622 (trusted 0.517). Use the *trusted* columns to compare retrievers in later phases. Test-split graded numbers are deferred to the Phase 10 final evaluation.

Promotion of `data/catalogue_metadata_backfilled.parquet` to production is pending user approval.

## Phase 3 — own-photo eval, baseline [classifier_category], half = holdout (2026-10-10T22:58:45)

Detections: `data/eval_own/baseline_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.589 [0.464, 0.732] |
| Item-category F1 (vs labelled counts) | 0.517 [0.413, 0.619] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.899 [0.841, 0.948] |
| P@5 (grade ≥ 2) | 0.924 [0.864, 0.980] |
| Success@1 (top-1 grade ≥ 2) | 0.880 [0.780, 0.960] |
| Mean grade@5 | 2.72 [2.57, 2.86] |
| judged@5 | 1.000 [1.000, 1.000] |
| Negatives rejected | 0.444 [0.111, 0.778] |
| … non_jewellery | 0.800 [0.400, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.560 [0.500, 0.625] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.576 [0.439, 0.667] | 0.857 [0.571, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.734 [0.643, 0.814] | 0.882 [0.765, 0.971] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.121 [0.000, 0.273] | 0.136 [0.000, 0.318] |

## Phase 3 — own-photo set prepared (2026-10-10)

`src/eval/prepare_own_eval.py` ran the current production pipeline (preprocess, then run_pipeline, top 5 of 20) on all 142 stumper photos and wrote `data/eval_own/{labels.csv, label.html, grades.csv, grade.html, baseline_detections.json, grade_items.json}`.
- labels: 60 core photos (all 13 mixed, plus folder-stratified picks, about 30 tune and 30 holdout), 65 optional, 17 negatives. Counts are prefilled from the detector; only rows with verified=1 are used.
- grades: 50 holdout items (8 from mixed photos, then earring 25, necklace 8, bracelet 9) × top-5 = 250 judgements.
- `src/eval/own_eval.py` scores any detections file of the same schema. Graded nDCG@5 is absolute (ideal = grade 3 at every rank). Unjudged results count as 0 and are reported as judged@5.
- Before labelling: out-of-scope jewellery negatives in holdout (ring, rings, watch, anklet) are rejected 0% of the time; non-jewellery negatives are rejected 80%.

## Phase 3 — own-photo eval, baseline [detector_category], half = holdout (2026-10-10T22:58:45)

Detections: `data/eval_own/baseline_detections.json`; category label: `detector_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.857 [0.750, 0.929] |
| Item-category F1 (vs labelled counts) | 0.753 [0.666, 0.830] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.899 [0.841, 0.948] |
| P@5 (grade ≥ 2) | 0.924 [0.864, 0.980] |
| Success@1 (top-1 grade ≥ 2) | 0.880 [0.780, 0.960] |
| Mean grade@5 | 2.72 [2.57, 2.86] |
| judged@5 | 1.000 [1.000, 1.000] |
| Negatives rejected | 0.444 [0.111, 0.778] |
| … non_jewellery | 0.800 [0.400, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.631 [0.500, 0.768] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.561 [0.439, 0.652] | 0.857 [0.571, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.775 [0.694, 0.849] | 0.941 [0.853, 1.000] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.712 [0.530, 0.879] | 0.727 [0.545, 0.909] |

## Phase 3 — own-photo eval, baseline [detector_category], half = tune (2026-10-10T22:58:46)

Detections: `data/eval_own/baseline_detections.json`; category label: `detector_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 61. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.934 [0.869, 0.984] |
| Category acc (first item vs folder) | 0.784 [0.667, 0.902] |
| Item-category F1 (vs labelled counts) | 0.627 [0.529, 0.722] |
| Count exact | 0.475 [0.344, 0.590] (over 0.11 / under 0.41) |
| Count MAE | 0.67 [0.49, 0.85] |
| Graded nDCG@5 (n=0, bad crops 0, unmatched 0) | – |
| P@5 (grade ≥ 2) | – |
| Success@1 (top-1 grade ≥ 2) | – |
| Mean grade@5 | – |
| judged@5 | – |
| Negatives rejected | 0.875 [0.625, 1.000] |
| … non_jewellery | 1.000 [1.000, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 2 | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | – |
| overlap | 14 | 0.071 [0.000, 0.214] | 0.478 [0.333, 0.618] | 0.714 [0.429, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.889 [0.889, 0.889] | – |
| pendant | 1 | 0.000 [0.000, 0.000] | 0.800 [0.800, 0.800] | – |
| blur | 4 | 0.500 [0.000, 1.000] | 0.417 [0.000, 0.833] | 0.667 [0.000, 1.000] |
| low_light | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | – |
| multi_item | 38 | 0.316 [0.184, 0.474] | 0.603 [0.483, 0.711] | 0.800 [0.667, 0.933] |
| no_tag | 20 | 0.750 [0.550, 0.900] | 0.717 [0.500, 0.900] | 0.789 [0.579, 0.947] |

## Phase 3 — summary: own-photo baseline (holdout, 2026-10-10)

The user verified 140 of 142 photos (58/60 core, 65/65 optional, 17/17 negatives) and graded 50 holdout items × 5. The detector's prefilled count was wrong on 64 of 123 positive photos. Holdout, current production:

| metric | classifier label (live) | detector label (shown in UI) |
|---|---|---|
| Category acc (first item) | 0.589 [0.464, 0.732] | 0.857 [0.750, 0.929] |
| Item-category F1 | 0.517 [0.413, 0.619] | 0.753 [0.666, 0.830] |
| Count exact / MAE | 0.629 [0.484, 0.758] / 0.47 | same |
| Undercount / overcount rate | 0.31 / 0.06 | same |
| Graded nDCG@5 / P@5 / Success@1 | 0.899 / 0.924 / 0.880 | same |
| Negatives rejected: non-jewellery / out-of-scope jewellery | 0.80 (n=5) / **0.00 (n=4)** | same |

Notes:
- Counting fails mainly by **undercounting**: overlap count-exact 0.09 (n=11), chain 0.00 (n=4). That supports the NMS-merge diagnosis for Phase 5.
- Graded retrieval is near ceiling (32/50 items have all five at grade 3; 211/250 judgements are 3), so it has little headroom to show gains. 6 wrong-category results were graded 3; to be confirmed with the user.
- hand_held, blur, low_light and pendant each have ≤1 holdout photo, so their per-tag numbers are not meaningful.
- The user decided to add no extra negatives for now. Out-of-scope jewellery rejection (0/4 holdout) stays the weakest point for Phase 8 calibration.

## Phase 4 — category identification (2026-10-11T18:20:46)

Same OWL-ViT boxes as the baseline. Items: tune 85, holdout 98 (item truth = folder for single-category photos; mixed photos count via F1 against labelled counts). Chosen on **tune**, reported on **holdout** (95% bootstrap CIs). kNN k=20, T=1.0 (chosen on tune). CLIP prompts `names+subtypes|plain` (chosen offline on train catalogue images). LR stacking leave-one-photo-out acc on tune: 0.773.

| signal(s) | tune item acc | holdout item acc | tune photo acc | holdout photo acc (first item) | tune item-cat F1 | holdout item-cat F1 |
|---|---|---|---|---|---|---|
| legacy | 0.627 | 0.697 [0.596, 0.787] | 0.588 | 0.589 [0.464, 0.732] | 0.455 | 0.517 [0.413, 0.619] |
| det | 0.720 | 0.843 [0.775, 0.921] | 0.784 | 0.857 [0.750, 0.929] | 0.627 | 0.753 [0.666, 0.830] |
| knn | 0.813 | 0.899 [0.831, 0.955] | 0.863 | 0.893 [0.804, 0.964] | 0.675 | 0.786 [0.712, 0.852] |
| clip | 0.573 | 0.685 [0.584, 0.787] | 0.608 | 0.768 [0.643, 0.875] | 0.523 | 0.652 [0.563, 0.741] |
| clf_v1 | 0.760 | 0.831 [0.753, 0.899] | 0.804 | 0.839 [0.750, 0.929] | 0.629 | 0.740 [0.661, 0.812] |
| clf_v2 | 0.813 | 0.955 [0.910, 0.989] | 0.882 | 0.964 [0.911, 1.000] | 0.681 | 0.829 [0.769, 0.880] |
| fused_grid | 0.853 | 0.933 [0.876, 0.978] | 0.902 | 0.929 [0.857, 0.982] | 0.715 | 0.821 [0.758, 0.876] |
| det+knn+clip+clf_v1 | 0.827 | 0.899 [0.831, 0.955] | 0.882 | 0.911 [0.839, 0.982] | 0.692 | 0.796 [0.723, 0.856] |
| det+knn+clf_v1+clf_v2 | 0.827 | 0.944 [0.888, 0.989] | 0.882 | 0.929 [0.857, 0.982] | 0.692 | 0.817 [0.750, 0.874] |
| det+clip+clf_v1+clf_v2 | 0.827 | 0.899 [0.831, 0.955] | 0.882 | 0.911 [0.839, 0.982] | 0.692 | 0.804 [0.734, 0.863] |
| knn+clip+clf_v1+clf_v2 | 0.827 | 0.933 [0.876, 0.978] | 0.863 | 0.929 [0.857, 0.982] | 0.690 | 0.821 [0.758, 0.876] |
| det+knn+clip+clf_v1+clf_v2 | 0.827 | 0.933 [0.876, 0.978] | 0.882 | 0.929 [0.857, 0.982] | 0.692 | 0.817 [0.750, 0.874] |
| knn+clip+clf_v2 **(winner)** | 0.840 | 0.933 [0.876, 0.978] | 0.902 | 0.946 [0.893, 1.000] | 0.708 | 0.813 [0.745, 0.872] |
| stack_lr | 0.893 | 0.865 [0.787, 0.944] | 0.941 | 0.857 [0.767, 0.946] | 0.729 | 0.755 [0.674, 0.825] |

Winner weights: `{'knn': 1.0, 'clip': 1.0, 'clf_v2': 1.0}`

ECE on stumper items (lower = better calibrated): legacy 0.276, knn 0.046, clip 0.116, clf_v1 0.075, clf_v2 0.069

Holdout confusion, legacy (rows = true; cols = earring / necklace / bracelet):

| true | earring | necklace | bracelet |
|---|---|---|---|
| earring | 62 | 0 | 0 |
| necklace | 14 | 0 | 0 |
| bracelet | 13 | 0 | 0 |

Holdout confusion, knn+clip+clf_v2 (rows = true; cols = earring / necklace / bracelet):

| true | earring | necklace | bracelet |
|---|---|---|---|
| earring | 57 | 3 | 2 |
| necklace | 1 | 13 | 0 |
| bracelet | 0 | 0 | 13 |

Per-tag holdout (item-category F1 / first-item acc):

| tag | n | legacy | det | winner |
|---|---|---|---|---|
| chain | 4 | 0.56 / 0.00 | 0.63 / 0.00 | 0.63 / 0.00 |
| overlap | 11 | 0.58 / 0.86 | 0.56 / 0.86 | 0.67 / 1.00 |
| hand_held | 1 | 0.00 / 0.00 | 0.67 / 1.00 | 0.67 / 1.00 |
| blur | 1 | 0.67 / 1.00 | 0.67 / 1.00 | 0.67 / 1.00 |
| multi_item | 40 | 0.73 / 0.88 | 0.77 / 0.94 | 0.75 / 0.94 |
| no_tag | 22 | 0.12 / 0.14 | 0.71 / 0.73 | 0.92 / 0.95 |

## Phase 3 — own-photo eval, phase4_fused_live [classifier_category], half = holdout (2026-10-11T19:31:22)

Detections: `data/eval_own/phase4_live_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.946 [0.893, 1.000] |
| Item-category F1 (vs labelled counts) | 0.813 [0.745, 0.872] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.894 [0.825, 0.949] |
| P@5 (grade ≥ 2) | 0.924 [0.856, 0.980] |
| Success@1 (top-1 grade ≥ 2) | 0.960 [0.900, 1.000] |
| Mean grade@5 | 2.68 [2.46, 2.85] |
| judged@5 | 1.000 [1.000, 1.000] |
| Negatives rejected | 0.444 [0.111, 0.778] |
| … non_jewellery | 0.800 [0.400, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.631 [0.500, 0.768] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.667 [0.606, 0.758] | 1.000 [1.000, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.752 [0.668, 0.824] | 0.941 [0.853, 1.000] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.924 [0.818, 1.000] | 0.955 [0.864, 1.000] |

## Phase 3 — own-photo eval, phase4_fused_live [classifier_category], half = tune (2026-10-11T13:45:07)

Detections: `data/eval_own/phase4_live_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 61. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.934 [0.869, 0.984] |
| Category acc (first item vs folder) | 0.902 [0.824, 0.980] |
| Item-category F1 (vs labelled counts) | 0.708 [0.620, 0.789] |
| Count exact | 0.475 [0.344, 0.590] (over 0.11 / under 0.41) |
| Count MAE | 0.67 [0.49, 0.85] |
| Graded nDCG@5 (n=0, bad crops 0, unmatched 0) | – |
| P@5 (grade ≥ 2) | – |
| Success@1 (top-1 grade ≥ 2) | – |
| Mean grade@5 | – |
| judged@5 | – |
| Negatives rejected | 0.875 [0.625, 1.000] |
| … non_jewellery | 1.000 [1.000, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 2 | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | – |
| overlap | 14 | 0.071 [0.000, 0.214] | 0.545 [0.405, 0.679] | 0.857 [0.571, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | – |
| pendant | 1 | 0.000 [0.000, 0.000] | 0.800 [0.800, 0.800] | – |
| blur | 4 | 0.500 [0.000, 1.000] | 0.667 [0.250, 1.000] | 1.000 [1.000, 1.000] |
| low_light | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | – |
| multi_item | 38 | 0.316 [0.184, 0.474] | 0.654 [0.553, 0.743] | 0.900 [0.800, 1.000] |
| no_tag | 20 | 0.750 [0.550, 0.900] | 0.817 [0.650, 0.950] | 0.895 [0.737, 1.000] |

## Phase 4 — summary (2026-10-11)

**Winner (chosen on tune): equal-weight average of kNN vote + CLIP zero-shot + retrained classifier (clf_v2).** Holdout (not used to choose):

| metric (holdout) | legacy (live) | detector label | **fused** |
|---|---|---|---|
| Item category acc | 0.697 | 0.843 | **0.933** [0.876, 0.978] |
| Photo category acc (first item) | 0.589 | 0.857 | **0.946** [0.893, 1.000] |
| Item-category F1 | 0.517 | 0.753 | **0.813** [0.745, 0.872] |
| necklace→earring / bracelet→earring errors | 14 / 13 | — | 1 / 0 |
| ECE on stumper items | 0.276 | — | clf_v2 0.069, kNN 0.046 |

- Root cause of the earring bias was the classifier's input normalisation, not class imbalance. Feeding v1 its training input raises val catalogue accuracy from 0.536 to 0.950. clf_v2 is trained on normalised embeddings with clean + augmented + detector-crop views (C=3, T=0.9), reaches 0.956 on val catalogue and has intercepts near zero.
- clf_v2 alone scores 0.955 on holdout items, which is higher than the fused winner, but it tied lower on tune (0.813 vs 0.840). Switching to it now would be choosing on holdout, so the winner stays as selected. The CIs overlap.
- The detector label gets weight 0 in the winner. It is still computed and returned (as `detector_category`) but no longer decides the category.
- Integration: `src/pipeline/category.py`, behind `config.CATEGORY_MODE` (still "legacy"). Live fused output matches the offline experiment on 196/196 items. Extra cost is about 28 ms per item (p50; mostly the CLIP image forward pass).
- Retrieval side effect on the own-photo set, holdout: Success@1 0.880 → 0.940. Graded nDCG@5 0.899 → 0.865, but 25 of 250 fused top-5 results were never graded (judged@5 0.90) and count as 0. Top-up grading page: `data/eval_own/grade_topup_phase4.html` (🙋).

## Phase 4 — promoted (2026-10-11)

- The user chose the **fused winner** (kNN + CLIP + clf_v2). `config.CATEGORY_MODE = "fused"`. The previous `config.py` is backed up in `artifacts_backup/20261011_193107/`; setting `"legacy"` restores v1 behaviour.
- The user graded the 25 new top-up results (`grades_topup_phase4.csv`). Holdout with complete grades (judged@5 = 1.00):

| own-photo holdout | baseline (legacy) | fused (live) |
|---|---|---|
| Graded nDCG@5 | 0.899 [0.841, 0.948] | 0.894 [0.825, 0.949] |
| P@5 (grade ≥ 2) | 0.924 | 0.924 |
| Success@1 | 0.880 [0.780, 0.960] | **0.960** [0.900, 1.000] |
| Category acc (first item) | 0.589 | **0.946** |

- API smoke test in fused mode: `/predict` returns the fused `detected_category` plus `category_confidence` and `detector_category`. The price filter still works.
- Catalogue Recall@k with the fused re-rank will be measured when the Phase 1 harness is extended in Phase 6, because the harness's `prod` variant still models the legacy re-rank.

## Phase 5 — detection and multi-item (2026-10-11T20:10:09)

Category of every box = Phase 4 fused category. Chosen on **tune**; holdout reported once. Objective: item-category F1 vs labelled counts, then count exact, then latency; detection recall must stay within 0.02 of baseline.

**1. Detector benchmark (tune, v1 prompts, threshold/NMS tuned per detector)**

| detector | thr / NMS IoU | det. recall | count exact | count MAE | item-cat F1 | under / over | latency p50 / p95 (s) |
|---|---|---|---|---|---|---|---|
| owlvit v1 config (baseline) | 0.1 / 0.3 | 0.934 | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.32 / 0.38 |
| owlvit **(chosen)** | 0.1 / 0.3 | 0.934 | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.32 / 0.38 |
| owlv2 | 0.12 / 0.3 | 0.918 | 0.279 | 1.87 | 0.616 | 0.23 / 0.49 | 3.55 / 3.82 |
| gdino | 0.3 / 0.3 | 0.967 | 0.443 | 0.64 | 0.732 | 0.26 / 0.30 | 5.33 / 6.20 |
| owlvit tuned (rejected: < margin) | 0.08 / 0.5 | 0.934 | 0.475 | 0.84 | 0.719 | 0.26 / 0.26 | 0.32 / 0.38 |

**2. Prompt ensembling on owlvit (tune)**

| prompts | thr / NMS | count exact | item-cat F1 | det. recall |
|---|---|---|---|---|
| current/max **(chosen)** | 0.1 / 0.3 | 0.475 | 0.708 | 0.934 |
| ensemble/max | 0.08 / 0.3 | 0.426 | 0.702 | 0.967 |
| ensemble/mean | 0.08 / 0.3 | 0.492 | 0.697 | 0.934 |
| ensemble/max/+jewellery | 0.12 / 0.3 | 0.443 | 0.707 | 1.000 |
| ensemble/mean/+jewellery | 0.2 / 0.3 | 0.443 | 0.712 | 0.967 |

**3. Post-processing rule ablation (tune)**

| config | count exact | count MAE | item-cat F1 | under / over | overlap count exact | chain count exact |
|---|---|---|---|---|---|---|
| (none) | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| +pair_protect | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| +group_drop | 0.508 | 0.62 | 0.715 | 0.41 / 0.08 | 0.07 (n=14) | 0.00 (n=2) |
| +chain_merge | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| +min_box | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| +nms_per_class | 0.475 | 0.67 | 0.713 | 0.38 / 0.15 | 0.14 (n=14) | 0.00 (n=2) |
| +tiles | 0.459 | 0.74 | 0.700 | 0.34 / 0.20 | 0.14 (n=14) | 0.00 (n=2) |
| +tiles(thr 0.15) | 0.459 | 0.70 | 0.703 | 0.39 / 0.15 | 0.07 (n=14) | 0.00 (n=2) |
| +tiles(thr 0.2) | 0.443 | 0.70 | 0.702 | 0.41 / 0.15 | 0.00 (n=14) | 0.00 (n=2) |
| +verify(0.45) | 0.459 | 0.69 | 0.696 | 0.44 / 0.10 | 0.00 (n=14) | 0.00 (n=2) |
| +verify(0.55) | 0.492 | 0.66 | 0.702 | 0.44 / 0.07 | 0.00 (n=14) | 0.00 (n=2) |
| grid group_drop=True | 0.508 | 0.62 | 0.715 | 0.41 / 0.08 | 0.07 (n=14) | 0.00 (n=2) |
| grid verify_sim=0.45 | 0.459 | 0.69 | 0.696 | 0.44 / 0.10 | 0.00 (n=14) | 0.00 (n=2) |
| grid verify_sim=0.45, group_drop=True | 0.492 | 0.66 | 0.702 | 0.44 / 0.07 | 0.00 (n=14) | 0.00 (n=2) |
| grid verify_sim=0.55 | 0.492 | 0.66 | 0.702 | 0.44 / 0.07 | 0.00 (n=14) | 0.00 (n=2) |
| grid verify_sim=0.55, group_drop=True | 0.475 | 0.67 | 0.697 | 0.46 / 0.07 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True | 0.459 | 0.74 | 0.700 | 0.34 / 0.20 | 0.14 (n=14) | 0.00 (n=2) |
| grid tiles=True, group_drop=True | 0.492 | 0.67 | 0.708 | 0.36 / 0.15 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, verify_sim=0.45 | 0.459 | 0.72 | 0.697 | 0.39 / 0.15 | 0.14 (n=14) | 0.00 (n=2) |
| grid tiles=True, verify_sim=0.45, group_drop=True | 0.475 | 0.69 | 0.702 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, verify_sim=0.55 | 0.492 | 0.69 | 0.699 | 0.41 / 0.10 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, verify_sim=0.55, group_drop=True | 0.459 | 0.70 | 0.691 | 0.44 / 0.10 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15 | 0.459 | 0.70 | 0.703 | 0.39 / 0.15 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15, group_drop=True | 0.475 | 0.67 | 0.706 | 0.41 / 0.11 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15, verify_sim=0.45 | 0.475 | 0.69 | 0.699 | 0.41 / 0.11 | 0.14 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15, verify_sim=0.45, group_drop=True | 0.492 | 0.67 | 0.702 | 0.43 / 0.08 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15, verify_sim=0.55 | 0.492 | 0.67 | 0.700 | 0.43 / 0.08 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.15, verify_sim=0.55, group_drop=True | 0.459 | 0.70 | 0.691 | 0.46 / 0.08 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2 | 0.443 | 0.70 | 0.702 | 0.41 / 0.15 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2, group_drop=True | 0.475 | 0.66 | 0.708 | 0.41 / 0.11 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2, verify_sim=0.45 | 0.459 | 0.69 | 0.698 | 0.43 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2, verify_sim=0.45, group_drop=True | 0.492 | 0.66 | 0.704 | 0.43 / 0.08 | 0.07 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2, verify_sim=0.55 | 0.475 | 0.67 | 0.699 | 0.44 / 0.08 | 0.00 (n=14) | 0.00 (n=2) |
| grid tiles=True, tile_score_thr=0.2, verify_sim=0.55, group_drop=True | 0.459 | 0.69 | 0.694 | 0.46 / 0.08 | 0.00 (n=14) | 0.00 (n=2) |
| FINAL (base, no rule beat the margin) | 0.475 | 0.67 | 0.708 | 0.41 / 0.11 | 0.07 (n=14) | 0.00 (n=2) |

Final config: `{'detector': 'owlvit', 'query_set': 'current', 'agg': 'max', 'agnostic_queries': False, 'nms_per_class': False, 'score_thr': 0.1, 'tiles': False, 'tile_score_thr': None, 'verify_sim': None, 'min_area_frac': 0.0, 'nms_iou': 0.3, 'pair_protect': False, 'group_drop': False, 'chain_merge': False}`

**4. Holdout: baseline (v1 detector) vs final — both with the Phase 4 fused category**

| metric | baseline | final |
|---|---|---|
| Detection recall | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] |
| Count exact | 0.629 [0.484, 0.758] | 0.629 [0.484, 0.758] |
| Item-category F1 | 0.813 [0.745, 0.872] | 0.813 [0.745, 0.872] |
| Category acc (first item) | 0.946 [0.893, 1.000] | 0.946 [0.893, 1.000] |
| Count MAE / under / over | 0.47 / 0.31 / 0.06 | 0.47 / 0.31 / 0.06 |
| Graded nDCG@5 | 0.894 [0.825, 0.950] | 0.894 [0.825, 0.950] |
| P@5 | 0.924 [0.856, 0.980] | 0.924 [0.856, 0.980] |
| Success@1 | 0.960 [0.900, 1.000] | 0.960 [0.900, 1.000] |
| judged@5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Graded items unmatched (no box IoU ≥ 0.5) | 0 | 0 |
| Negatives rejected | 0.444 [0.111, 0.778] | 0.444 [0.111, 0.778] |
| Latency p50 / p95 (detect + post-process, s) | 0.32 / 0.38 | 0.32 / 0.38 |

Per-tag holdout (count exact / item-category F1):

| tag | n | baseline | final |
|---|---|---|---|
| chain | 4 | 0.00 / 0.63 | 0.00 / 0.63 |
| overlap | 11 | 0.09 / 0.67 | 0.09 / 0.67 |
| hand_held | 1 | 0.00 / 0.67 | 0.00 / 0.67 |
| multi_item | 40 | 0.47 / 0.75 | 0.47 / 0.75 |
| no_tag | 22 | 0.91 / 0.92 | 0.91 / 0.92 |

## Phase 5 — summary (2026-10-11)

**Outcome: no detection change is adopted. Production stays on OWL-ViT with the v1 prompts, threshold 0.1, NMS 0.3.** No candidate beat the v1 config on tune by the 0.02 F1 margin (about one photo).
- **Diagnosis (tune):** 20 of 25 undercounted photos are a recall problem, not a merge problem. The missing items never get a box above threshold before NMS. The typical case is earrings lying inside a necklace's loop (m5, m6, m7, m10), where the necklace box covers them and they get no box of their own. Only 5 of 25 are suppression cases, so the NMS rules (pair protection, chain merge) never triggered on these photos.
- **Detectors:** Grounding DINO has the best tune F1 (0.732 vs 0.708 for v1 OWL-ViT) but is about 16× slower (5.3 s vs 0.32 s per image, CPU, Low Power Mode). Its gain over tuned OWL-ViT (+0.013) is below the margin. OWLv2 overcounts heavily (over 0.49, F1 0.616).
- **Prompt ensembling:** the ensembles raise recall (up to 1.000 with a "jewellery" phrase) but not F1 (0.697–0.712 vs 0.708).
- **Tiled (SAHI-style) detection:** cuts undercounting (0.41 → 0.34) but adds about as much overcounting (0.11 → 0.20). Net F1 0.700.
- **Retrieval verification** (drop a box whose crop's top-1 catalogue similarity is below 0.45 or 0.55) removes spurious boxes but also real ones. F1 ≤ 0.702.
- **Why the margin matters:** in a dry run without it, the tune-best config (threshold 0.08, NMS 0.5, group_drop, nms_per_class; tune F1 +0.018) was **worse on holdout**: count exact 0.629 → 0.548, F1 0.813 → 0.777, Success@1 0.960 → 0.800, with 8 graded items losing their box.
- Holdout is therefore unchanged: count exact 0.629, item-category F1 0.813, overlap count exact 0.09 (n=11), chain 0.00 (n=4).
- Built and kept for later use: `src/pipeline/detectors.py` (owlvit / owlv2 / gdino behind one interface), `src/pipeline/postprocess.py` (all rules; its defaults reproduce v1 exactly, 196/196 boxes), raw caches `data/eval_own/phase5_raw_*.json`, crop cache, and `src/eval/detection_bench.py`.

## Phase 6 — embedding variants (2026-10-11T23:05:31, split = val)

Same-listing protocol, exact search, query image excluded. Primary = mean of MRR(clean, all 502) and MRR(combined phone-photo augmentation, 300 subset), chosen on val. *fused* = with the Phase 4 fused re-rank (production). Graded nDCG uses trusted labels only (title/variant, no image-model fills).

HNSW check (variant A, Phase 1 numbers): R@1 raw 0.297 vs exact 0.297.

| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | R@1 combined aug | MRR combined aug | index MB | query embed s | search ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A dino **(winner)** | 0.2636 | 0.2636 | 0.297 | 0.424 | 0.472 | 0.351 | 0.362 | 0.710 | 0.959 | 0.140 | 0.176 | 41 | 0.123 | 0.6 |
| B dino_nobg | 0.2327 | 0.2347 | 0.293 | 0.406 | 0.438 | 0.343 | 0.353 | 0.670 | 0.908 | 0.103 | 0.123 | 41 | 0.441 | 0.6 |
| C clip | 0.1628 | 0.1652 | 0.239 | 0.329 | 0.380 | 0.284 | 0.287 | 0.673 | 0.949 | 0.030 | 0.042 | 27 | 0.035 | 0.3 |
| D dino+clip w=0.3 | 0.2281 | 0.2289 | 0.273 | 0.410 | 0.468 | 0.336 | 0.346 | 0.711 | 0.968 | 0.087 | 0.120 | 68 | 0.158 | 1.0 |
| D dino+clip w=0.4 | 0.2380 | 0.2393 | 0.275 | 0.422 | 0.472 | 0.341 | 0.353 | 0.714 | 0.967 | 0.097 | 0.135 | 68 | 0.158 | 1.1 |
| D dino+clip w=0.5 | 0.2480 | 0.2482 | 0.279 | 0.424 | 0.478 | 0.345 | 0.357 | 0.715 | 0.967 | 0.110 | 0.151 | 68 | 0.158 | 1.1 |
| D dino+clip w=0.6 | 0.2588 | 0.2590 | 0.285 | 0.428 | 0.478 | 0.348 | 0.361 | 0.713 | 0.964 | 0.133 | 0.169 | 68 | 0.158 | 1.1 |
| D dino+clip w=0.7 | 0.2633 | 0.2636 | 0.293 | 0.426 | 0.478 | 0.353 | 0.363 | 0.714 | 0.965 | 0.140 | 0.174 | 68 | 0.158 | 1.2 |
| D dino+clip w=0.8 | 0.2620 | 0.2637 | 0.293 | 0.424 | 0.478 | 0.353 | 0.363 | 0.713 | 0.962 | 0.133 | 0.171 | 68 | 0.158 | 1.3 |
| D dino+clip w=0.9 | 0.2623 | 0.2641 | 0.293 | 0.422 | 0.476 | 0.350 | 0.360 | 0.712 | 0.961 | 0.137 | 0.174 | 68 | 0.158 | 1.5 |
| B+C nobg+clip w=0.3 | 0.2074 | 0.2109 | 0.269 | 0.410 | 0.448 | 0.329 | 0.344 | 0.702 | 0.960 | 0.060 | 0.085 | 68 | 0.475 | 1.5 |
| B+C nobg+clip w=0.4 | 0.2188 | 0.2202 | 0.281 | 0.410 | 0.444 | 0.338 | 0.350 | 0.704 | 0.960 | 0.073 | 0.100 | 68 | 0.475 | 1.6 |
| B+C nobg+clip w=0.5 | 0.2270 | 0.2281 | 0.287 | 0.412 | 0.448 | 0.343 | 0.355 | 0.704 | 0.955 | 0.083 | 0.111 | 68 | 0.475 | 2.0 |
| B+C nobg+clip w=0.6 | 0.2305 | 0.2315 | 0.289 | 0.412 | 0.450 | 0.345 | 0.356 | 0.699 | 0.948 | 0.090 | 0.116 | 68 | 0.475 | 2.4 |
| B+C nobg+clip w=0.7 | 0.2287 | 0.2293 | 0.283 | 0.406 | 0.448 | 0.341 | 0.351 | 0.691 | 0.939 | 0.090 | 0.116 | 68 | 0.475 | 2.4 |
| B+C nobg+clip w=0.8 | 0.2306 | 0.2321 | 0.283 | 0.410 | 0.448 | 0.341 | 0.353 | 0.684 | 0.928 | 0.097 | 0.121 | 68 | 0.475 | 2.4 |
| B+C nobg+clip w=0.9 | 0.2327 | 0.2343 | 0.285 | 0.412 | 0.444 | 0.341 | 0.354 | 0.678 | 0.920 | 0.103 | 0.124 | 68 | 0.475 | 2.3 |
| E A dino | mean | 0.0904 | 0.0916 | 0.076 | 0.149 | 0.207 | 0.114 | 0.114 | 0.546 | 0.869 | 0.043 | 0.066 | 41 | 0.123 | 1.4 |
| E A dino | maxmean | 0.1933 | 0.1936 | 0.203 | 0.315 | 0.371 | 0.258 | 0.263 | 0.634 | 0.915 | 0.090 | 0.129 | 41 | 0.123 | 1.3 |

Winner `A dino` clean R@1 0.297 [0.257, 0.335], MRR 0.351 [0.311, 0.387]; A dino R@1 0.297 [0.257, 0.335].

**Per augmentation, MRR (no re-rank)**

| aug | A dino | B dino_nobg | C clip |
|---|---|---|---|
| clean | 0.351 | 0.343 | 0.284 |
| blur | 0.180 | 0.062 | 0.064 |
| low_light | 0.341 | 0.296 | 0.165 |
| bright_contrast | 0.348 | 0.274 | 0.171 |
| rotate | 0.319 | 0.275 | 0.212 |
| crop_zoom | 0.333 | 0.307 | 0.240 |
| jpeg | 0.372 | 0.321 | 0.254 |
| busy_bg | 0.114 | 0.142 | 0.035 |
| perspective | 0.345 | 0.311 | 0.248 |
| combined | 0.176 | 0.123 | 0.042 |

## Phase 6 — summary (2026-10-11)

**Outcome: DINOv2-base raw (variant A, the current production embedding) stays. Nothing to promote.** Chosen on val; the primary score is the mean of MRR on clean queries and on the combined phone-photo augmentation.

| variant | primary | R@1 clean | MRR clean | MRR combined aug | MRR busy_bg | MRR blur | query embed |
|---|---|---|---|---|---|---|---|
| **A DINOv2** | **0.264** | 0.297 | 0.351 | **0.176** | 0.114 | **0.180** | 0.12 s |
| B DINOv2, background removed | 0.233 | 0.293 | 0.343 | 0.123 | **0.142** | 0.062 | 0.44 s |
| C CLIP B/32 | 0.163 | 0.239 | 0.284 | 0.042 | 0.035 | 0.064 | 0.04 s |
| D DINOv2+CLIP (best w=0.7) | 0.263 | 0.293 | 0.353 | 0.174 | — | — | 0.16 s |
| E DINOv2, listing = mean of images | 0.090 | 0.076 | 0.114 | 0.066 | — | — | 0.12 s |

- **B (background removal)** helps only the case it targets: busy backgrounds, 0.114 → 0.142. It hurts every other condition, most of all blur (0.180 → 0.062, because u2net fails on blurred images). It also lowers category P@5 (0.959 → 0.908) and costs 3.6× the query time. It is not worth it as a global step. A gated version (remove the background only when the background is busy) is possible but unproven.
- **CLIP** is far less robust to phone conditions than DINOv2. Fusing it with DINOv2 never beats DINOv2 alone.
- **Listing aggregation:** mean-of-images is far worse than max, because listings mix model shots and product shots. Max (v1) is correct.
- **The Phase 4 fused re-rank** changes catalogue retrieval by less than 0.003 primary for every variant. Its benefit appears on own photos instead (Success@1 0.88 → 0.96).
- Exact search reproduces Phase 1's HNSW numbers (R@1 0.297 both), so the HNSW approximation costs nothing here.
- Artifacts built (not promoted): `artifacts/dinov2_nobg/`, `artifacts/clip_b32/`, and query feature caches in `artifacts/eval_cache/val_*.npz`, which Phase 7 reuses.
- The weakest conditions remain busy backgrounds and blur. Phase 7's head is trained on augmented views (busy_bg and blur included) to address exactly these.

## Phase 7 — contrastive projection head (2026-10-12T09:42:52)

Head: linear, tau=0.1, hard negatives=True, best step 2700 (chosen on val). Trained on train-split listings only. Projection adds 0.10 ms per query; index 41 MB, dim 768.

**Catalogue val (exact search, same-listing protocol)**

| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | colour P@5 (trusted) | subtype P@5 (trusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A frozen | 0.2636 | 0.2636 | 0.297 [0.257, 0.335] | 0.424 | 0.472 | 0.351 [0.311, 0.387] | 0.362 | 0.710 | 0.959 | 0.470 | 0.522 |
| FT | 0.3830 | 0.3877 | 0.408 [0.365, 0.448] | 0.576 | 0.647 | 0.479 [0.440, 0.517] | 0.491 | 0.707 | 0.878 | 0.544 | 0.509 |

Per augmentation (val, R@1 / MRR, no re-rank):

| aug | A frozen | FT |
|---|---|---|
| clean | 0.297 / 0.351 | 0.408 / 0.479 |
| blur | 0.137 / 0.180 | 0.260 / 0.319 |
| low_light | 0.283 / 0.341 | 0.377 / 0.459 |
| bright_contrast | 0.293 / 0.348 | 0.387 / 0.458 |
| rotate | 0.260 / 0.319 | 0.377 / 0.458 |
| crop_zoom | 0.277 / 0.333 | 0.383 / 0.466 |
| jpeg | 0.323 / 0.372 | 0.400 / 0.475 |
| busy_bg | 0.073 / 0.114 | 0.217 / 0.290 |
| perspective | 0.280 / 0.345 | 0.380 / 0.455 |
| combined | 0.140 / 0.176 | 0.220 / 0.287 |

Per category (clean, non-set, R@1 / MRR):

| category | A frozen | FT |
|---|---|---|
| earring | 0.288 / 0.343 | 0.416 / 0.475 |
| necklace | 0.222 / 0.279 | 0.315 / 0.413 |
| bracelet | 0.394 / 0.442 | 0.452 / 0.527 |

**Own photos (production boxes; retrieval and fused category from each variant's own neighbours)**

| metric | half | A frozen | FT |
|---|---|---|---|
| Category acc (first item) | tune | 0.902 [0.824, 0.980] | 0.882 [0.784, 0.961] |
| Item-category F1 | tune | 0.712 [0.626, 0.792] | 0.682 [0.601, 0.761] |
| Category acc (first item) | holdout | 0.946 [0.893, 1.000] | 0.946 [0.893, 1.000] |
| Item-category F1 | holdout | 0.813 [0.745, 0.872] | 0.809 [0.741, 0.869] |
| Graded nDCG@5 | holdout | 0.891 [0.823, 0.948] | 0.249 [0.181, 0.317] |
| P@5 | holdout | 0.920 [0.852, 0.976] | 0.220 [0.160, 0.280] |
| Success@1 | holdout | 0.960 [0.900, 1.000] | 0.400 [0.280, 0.540] |
| judged@5 | holdout | 0.996 [0.988, 1.000] | 0.240 [0.184, 0.296] |

**Phase 7 notes (2026-10-12):**
- Training grid (val primary; frozen = 0.2636): linear τ=0.05 0.379 / with hard negatives 0.375; **linear τ=0.1 0.380 / with hard negatives 0.383 (chosen)**; MLP 0.358–0.373. The linear head (identity-initialised) beats the residual MLP in every setting. Most linear runs peaked near the 3,000-step cap.
- With the fused re-rank (production), category P@5 on val clean is A 0.972 vs FT 0.947. On combined-aug queries it is A 0.841 vs FT 0.845. R@1 is A 0.295 vs FT 0.412. The head trades a small amount of category purity for much better identity retrieval; the fused re-rank recovers most of that purity.
- Own-photo graded retrieval for FT is not yet measurable: only 24% of FT's top-5 on holdout items had been graded (judged@5 0.24). Top-up page: `data/eval_own/grade_topup_phase7.html` (🙋).
- Own-photo category accuracy is unchanged within noise (holdout 0.946 vs 0.946; tune 0.902 vs 0.882).
- Test split not run yet (it runs once, after the user has seen val).

## Phase 7 — test split, evaluated once (2026-10-12T11:16:39)

**Catalogue test (exact search, same-listing protocol)**

| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | colour P@5 (trusted) | subtype P@5 (trusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A frozen | 0.2848 | 0.2847 | 0.316 [0.276, 0.357] | 0.440 | 0.499 | 0.378 [0.339, 0.416] | 0.385 | 0.725 | 0.962 | 0.510 | 0.568 |
| FT | 0.4064 | 0.4137 | 0.434 [0.391, 0.477] | 0.606 | 0.686 | 0.517 [0.477, 0.555] | 0.527 | 0.720 | 0.868 | 0.585 | 0.552 |

Per augmentation (test, R@1 / MRR, no re-rank):

| aug | A frozen | FT |
|---|---|---|
| clean | 0.316 / 0.378 | 0.434 / 0.517 |
| blur | 0.133 / 0.183 | 0.233 / 0.319 |
| low_light | 0.300 / 0.355 | 0.410 / 0.497 |
| bright_contrast | 0.280 / 0.345 | 0.397 / 0.490 |
| rotate | 0.277 / 0.341 | 0.413 / 0.494 |
| crop_zoom | 0.280 / 0.346 | 0.410 / 0.499 |
| jpeg | 0.313 / 0.366 | 0.430 / 0.518 |
| busy_bg | 0.083 / 0.131 | 0.223 / 0.304 |
| perspective | 0.303 / 0.361 | 0.410 / 0.500 |
| combined | 0.143 / 0.192 | 0.213 / 0.296 |

Per category (clean, non-set, R@1 / MRR):

| category | A frozen | FT |
|---|---|---|
| earring | 0.361 / 0.416 | 0.452 / 0.529 |
| necklace | 0.257 / 0.299 | 0.371 / 0.458 |
| bracelet | 0.305 / 0.392 | 0.467 / 0.558 |

## Phase 3 — own-photo eval, phase7_dino [classifier_category], half = holdout (2026-10-12T11:16:51)

Detections: `data/eval_own/phase7_dino_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.946 [0.893, 1.000] |
| Item-category F1 (vs labelled counts) | 0.813 [0.745, 0.872] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.872 [0.803, 0.929] |
| P@5 (grade ≥ 2) | 0.912 [0.840, 0.968] |
| Success@1 (top-1 grade ≥ 2) | 0.960 [0.900, 1.000] |
| Mean grade@5 | 2.61 [2.40, 2.78] |
| judged@5 | 0.996 [0.988, 1.000] |
| Negatives rejected | 0.444 [0.111, 0.778] |
| … non_jewellery | 0.800 [0.400, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.631 [0.500, 0.768] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.667 [0.606, 0.758] | 1.000 [1.000, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.752 [0.668, 0.824] | 0.941 [0.853, 1.000] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.924 [0.818, 1.000] | 0.955 [0.864, 1.000] |

## Phase 3 — own-photo eval, phase7_ft [classifier_category], half = holdout (2026-10-12T11:16:51)

Detections: `data/eval_own/phase7_ft_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.946 [0.893, 1.000] |
| Item-category F1 (vs labelled counts) | 0.809 [0.741, 0.869] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.612 [0.543, 0.677] |
| P@5 (grade ≥ 2) | 0.668 [0.592, 0.744] |
| Success@1 (top-1 grade ≥ 2) | 0.760 [0.640, 0.880] |
| Mean grade@5 | 1.78 [1.57, 1.97] |
| judged@5 | 1.000 [1.000, 1.000] |
| Negatives rejected | 0.444 [0.111, 0.778] |
| … non_jewellery | 0.800 [0.400, 1.000] |
| … out_of_scope_jewellery | 0.000 [0.000, 0.000] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.631 [0.500, 0.768] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.667 [0.606, 0.758] | 1.000 [1.000, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.746 [0.662, 0.820] | 0.941 [0.853, 1.000] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.924 [0.818, 1.000] | 0.955 [0.864, 1.000] |

## Phase 7 — own-photo result and decision (2026-10-12)

The user top-up graded 190 FT results. Holdout, complete grades (judged@5 ≥ 0.996), production boxes, fused category and re-rank from each variant's own neighbours:

| own photos, holdout (50 items) | frozen DINOv2 | FT head |
|---|---|---|
| Graded nDCG@5 | **0.872** [0.803, 0.929] | 0.612 [0.543, 0.677] |
| P@5 (grade ≥ 2) | **0.912** | 0.668 |
| Success@1 | **0.960** | 0.760 |
| Wrong-category share of top-5 | **8%** | 20% |
| Category acc (first item) | 0.946 | 0.946 |

Paired, FT − frozen: nDCG@5 −0.261 [−0.333, −0.193]; Success@1 −0.20 [−0.32, −0.08]. 44 of 50 items worse.

**Test split (once):** R@1 0.316 → 0.434, MRR 0.378 → 0.517, every augmentation and category improved (table above). The catalogue gain is real on held-out listings.

**Interpretation:** the head learns *listing identity*, including the shop's photography style (background, model, lighting), because positives are other photos of the same listing. That is exactly what the same-listing protocol rewards. It is the wrong signal for this app's main use, where your piece is not in the catalogue and the goal is similar designs. FT pulls in more cross-category look-alikes (20% vs 8% of top-5).

**Caveat (grading confound):** the top-up page showed each listing's main photo, not its best-matching image (they differ for 68% of the 190 top-up results). FT results graded on the original pages (matched image shown) average 2.43, against 1.57 on the top-up page, so some of the drop may come from presentation. The wrong-category share (20% vs 8%) is independent of the thumbnail shown, and wrong-category results score 0.02 on average, so the direction of the result holds.

**Decision: FT is not promoted.** `artifacts/dinov2_ft/` is kept as a documented negative result for in-the-wild use. Production retrieval stays on frozen DINOv2. Lesson for the writeup: same-listing Recall@k is not aligned with user-facing similar-item quality, so own-photo graded evaluation is necessary.

Note: the holdout half has now been used to evaluate FT once. Any further retrieval variant must be chosen without holdout (on tune grades or on catalogue val) to avoid turning holdout into a tuning set.

## Phase 7 — contrastive projection head [ftaug] (2026-10-12T11:52:50)

Head: linear, tau=0.05, hard negatives=True, best step 1200 (chosen on val). Trained on train-split listings only. Projection adds 0.08 ms per query; index 41 MB, dim 768.

**Catalogue val (exact search, same-listing protocol)**

| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | colour P@5 (trusted) | subtype P@5 (trusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A frozen | 0.2636 | 0.2636 | 0.297 [0.257, 0.335] | 0.424 | 0.472 | 0.351 [0.311, 0.387] | 0.362 | 0.710 | 0.959 | 0.470 | 0.522 |
| FT | 0.3079 | 0.3088 | 0.321 [0.279, 0.361] | 0.450 | 0.508 | 0.381 [0.341, 0.419] | 0.389 | 0.696 | 0.923 | 0.508 | 0.507 |

Per augmentation (val, R@1 / MRR, no re-rank):

| aug | A frozen | FT |
|---|---|---|
| clean | 0.297 / 0.351 | 0.321 / 0.381 |
| blur | 0.137 / 0.180 | 0.213 / 0.256 |
| low_light | 0.283 / 0.341 | 0.293 / 0.357 |
| bright_contrast | 0.293 / 0.348 | 0.320 / 0.373 |
| rotate | 0.260 / 0.319 | 0.270 / 0.345 |
| crop_zoom | 0.277 / 0.333 | 0.317 / 0.375 |
| jpeg | 0.323 / 0.372 | 0.317 / 0.379 |
| busy_bg | 0.073 / 0.114 | 0.167 / 0.223 |
| perspective | 0.280 / 0.345 | 0.293 / 0.361 |
| combined | 0.140 / 0.176 | 0.187 / 0.235 |

Per category (clean, non-set, R@1 / MRR):

| category | A frozen | FT |
|---|---|---|
| earring | 0.288 / 0.343 | 0.313 / 0.364 |
| necklace | 0.222 / 0.279 | 0.241 / 0.321 |
| bracelet | 0.394 / 0.442 | 0.404 / 0.461 |

**Own photos (production boxes; retrieval and fused category from each variant's own neighbours)**

| metric | half | A frozen | FT |
|---|---|---|---|
| Category acc (first item) | tune | 0.902 [0.824, 0.980] | 0.882 [0.784, 0.961] |
| Item-category F1 | tune | 0.712 [0.626, 0.792] | 0.690 [0.608, 0.768] |

## Phase 7b — augmentation-only head (FTaug) and grading-rule correction (2026-10-12)

**Grading rule correction.** The Phase 7 top-up page re-showed 60 earlier-graded results (pre-selected) using the listing's main photo rather than the matched image. 13 of the 60 were changed, mostly 3 → 2. `own_eval.load_grades` now applies **first grade wins**: the original judgement, made while viewing the matched image, is kept, and later pages only add results not graded before. Re-scored holdout: baseline 0.899 (unchanged from Phase 3/4), Phase 7 frozen-exact 0.901, Phase 7 FT 0.638 (Success@1 0.960 vs 0.780). The FT decision stands.

**FTaug** = the same linear head, but positives are only augmented views of the *same image* (no cross-image listing positives), so it cannot learn the shop-style shortcut. Grid chosen on val: linear, τ=0.05, hard negatives, step 1,200.

| val (catalogue) | frozen | FT (listing positives) | **FTaug** |
|---|---|---|---|
| primary | 0.264 | 0.383 | 0.308 |
| R@1 / MRR clean | 0.297 / 0.351 | 0.408 / 0.479 | 0.321 / 0.381 |
| MRR busy_bg / blur / combined | 0.114 / 0.180 / 0.176 | 0.290 / 0.319 / 0.287 | 0.223 / 0.256 / 0.235 |
| cat P@5 (non-set, no re-rank) | 0.959 | 0.878 | 0.923 |

- Own-photo tune category: 0.902 (frozen) vs 0.882 (FTaug), within noise. The holdout half has not been used for FTaug.
- 🙋 Blind pooled tune grading page (frozen ∪ FTaug top-5, shuffled, no ranks or scores, matched-image thumbnails): `data/eval_own/grade_pool_phase7b.html`, 36 tune items, 314 results. Decision rule, fixed in advance: FTaug goes to a single holdout run only if its tune graded nDCG@5 is not lower than frozen's (paired difference ≥ 0) **and** its wrong-category share is not higher.

**Phase 7b tune result (blind pooled grading, 35 tune items; 1 bad crop excluded; judged@5 = 1.00 for both):**

| own photos, tune | frozen | FTaug | paired diff [95% CI] |
|---|---|---|---|
| Graded nDCG@5 | **0.635** | 0.538 | −0.096 [−0.151, −0.043] |
| P@5 (grade ≥ 2) | **0.754** | 0.646 | −0.109 [−0.189, −0.034] |
| Success@1 | **0.771** | 0.714 | −0.057 [−0.200, +0.086] |
| Wrong-category share of top-5 | **13.6%** | 25.7% | — |

FTaug was better on 7 items, the same on 5 and worse on 23. **The pre-set rule fails on both conditions, so FTaug gets no holdout run.**

**Phase 7 closing decision:** production retrieval stays on frozen DINOv2. Both trained heads improve the catalogue same-listing metrics (test R@1 +0.118 for FT; val R@1 +0.024 for FTaug, with large gains on busy_bg and blur) but make similar-item recommendations on real photos worse. The common failure is more cross-category look-alikes in the top-5 (frozen 8% / 14% vs FT 20% and FTaug 26%). Instance-discrimination training on catalogue images sharpens instance features at the cost of the category and design-family structure that frozen DINOv2 already has. Same-listing Recall@k cannot see this; own-photo graded evaluation can. A head with an explicit category or design-family term would be the next thing to try. It is not attempted here, so that holdout is not used again.

Note: the tune absolute numbers (0.635) are lower than the holdout baseline (0.899) because this pooled page was graded blind with matched-image thumbnails and about 9 results per item. Only the paired frozen-vs-FTaug comparison is meaningful here.

## Phase 8 — open-set calibration (2026-10-12T12:16:20)

Thresholds fixed by a **false-reject budget of 5% on tune positive photos** (half each for similarity and the CLIP out-of-scope check); false-accept rate reported on negatives (holdout has 9 negatives, so CIs are wide). Out-of-scope prompts were written down a priori, not tuned.

Calibrated: LOW_MATCH_THRESHOLD **0.475**, OOS_THRESHOLD **0.813**, HIGH_MATCH_THRESHOLD **0.74**, MARGIN_THRESHOLD **0.0** (v1: 0.45 / none / 0.65 / 0.03).

| decision rule | half | false-reject rate (positive photos) | false-accept rate (negative photos) | FAR non-jewellery | FAR out-of-scope jewellery |
|---|---|---|---|---|---|
| v1 (LOW=0.45, no OOS check) | tune | 0.018 [0.000, 0.053] (n=57) | 0.125 [0.000, 0.375] (n=8) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| v1 (LOW=0.45, no OOS check) | holdout | 0.000 [0.000, 0.000] (n=63) | 0.556 [0.222, 0.889] (n=9) | 0.200 [0.000, 0.600] | 1.000 [1.000, 1.000] |
| similarity only (calibrated T_low) | tune | 0.035 [0.000, 0.088] (n=57) | 0.125 [0.000, 0.375] (n=8) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| similarity only (calibrated T_low) | holdout | 0.000 [0.000, 0.000] (n=63) | 0.556 [0.222, 0.889] (n=9) | 0.200 [0.000, 0.600] | 1.000 [1.000, 1.000] |
| OOS check only (calibrated tau) | tune | 0.035 [0.000, 0.088] (n=57) | 0.250 [0.000, 0.500] (n=8) | 0.143 [0.000, 0.429] | 1.000 [1.000, 1.000] |
| OOS check only (calibrated tau) | holdout | 0.063 [0.016, 0.127] (n=63) | 0.333 [0.000, 0.667] (n=9) | 0.000 [0.000, 0.000] | 0.750 [0.250, 1.000] |
| calibrated T_low + OOS check | tune | 0.053 [0.000, 0.123] (n=57) | 0.125 [0.000, 0.375] (n=8) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| calibrated T_low + OOS check | holdout | 0.063 [0.016, 0.127] (n=63) | 0.333 [0.000, 0.667] (n=9) | 0.000 [0.000, 0.000] | 0.750 [0.250, 1.000] |

Closest-matches tier: result score ≥ **0.74** (margin rule: 0.0). Criterion: P(grade ≥ 2) ≥ 0.9 over every graded result on tune photos (n=173). Tune at threshold: (0.90625, 32); holdout check: (1.0, 64) (precision, n). v1 threshold 0.65: {'tune': (0.7549019607843137, 102), 'holdout': (0.9379310344827586, 145)}.

FAR/FRR curve (T_low sweep; with the calibrated OOS check):

| T_low | tune FRR | tune FAR | holdout FRR | holdout FAR | holdout FAR (similarity only) |
|---|---|---|---|---|---|
| 0.3 | 0.035 | 0.250 | 0.063 | 0.333 | 0.667 |
| 0.34 | 0.035 | 0.250 | 0.063 | 0.333 | 0.667 |
| 0.38 | 0.035 | 0.250 | 0.063 | 0.333 | 0.667 |
| 0.42 | 0.035 | 0.250 | 0.063 | 0.333 | 0.667 |
| 0.46 | 0.053 | 0.125 | 0.063 | 0.333 | 0.556 |
| 0.5 | 0.070 | 0.125 | 0.063 | 0.333 | 0.556 |
| 0.54 | 0.088 | 0.125 | 0.079 | 0.333 | 0.556 |
| 0.58 | 0.158 | 0.125 | 0.127 | 0.333 | 0.556 |
| 0.62 | 0.228 | 0.125 | 0.206 | 0.333 | 0.333 |
| 0.66 | 0.246 | 0.125 | 0.397 | 0.222 | 0.222 |
| 0.7 | 0.421 | 0.000 | 0.524 | 0.222 | 0.222 |
| 0.74 | 0.684 | 0.000 | 0.683 | 0.111 | 0.111 |
| 0.78 | 0.807 | 0.000 | 0.794 | 0.111 | 0.111 |
| 0.82 | 0.895 | 0.000 | 0.810 | 0.000 | 0.000 |
| 0.86 | 0.930 | 0.000 | 0.905 | 0.000 | 0.000 |

## Phase 8 — MMR diversity (2026-10-12T12:17:09)

Catalogue val, production ranking (frozen DINOv2 + fused re-rank), MMR over the top 20. Rule: most diverse with nDCG@5 and R@5 >= 98% of lambda=1 on clean and combined-aug queries. **Chosen λ = 0.7.**

| λ | clean nDCG@5 | clean R@5 | clean graded (trusted) | clean cat P@5 | clean ILD@5 | clean near-dup pairs | combined nDCG@5 | combined R@5 | combined ILD@5 |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.363 | 0.428 | 0.717 | 0.972 | 0.219 | 0.139 | 0.183 | 0.223 | 0.359 |
| 0.9 | 0.366 | 0.428 | 0.717 | 0.972 | 0.229 | 0.125 | 0.186 | 0.227 | 0.403 |
| 0.8 | 0.371 | 0.436 | 0.716 | 0.971 | 0.240 | 0.104 | 0.193 | 0.240 | 0.434 |
| 0.7 **(chosen)** | 0.379 | 0.446 | 0.715 | 0.968 | 0.256 | 0.084 | 0.185 | 0.227 | 0.455 |
| 0.6 | 0.382 | 0.456 | 0.712 | 0.968 | 0.276 | 0.053 | 0.176 | 0.213 | 0.469 |
| 0.5 | 0.373 | 0.444 | 0.702 | 0.965 | 0.296 | 0.022 | 0.159 | 0.177 | 0.478 |

## Phase 3 — own-photo eval, phase8_live [classifier_category], half = holdout (2026-10-12T12:20:31)

Detections: `data/eval_own/phase8_live_detections.json`; category label: `classifier_category`. 95% bootstrap CIs. Labelled (verified) positive photos in this half: 62. Graded nDCG@5 is absolute (ideal = grade 3 at every rank).

| metric | value |
|---|---|
| Detection recall | 0.984 [0.953, 1.000] |
| Category acc (first item vs folder) | 0.946 [0.893, 1.000] |
| Item-category F1 (vs labelled counts) | 0.813 [0.745, 0.872] |
| Count exact | 0.629 [0.484, 0.758] (over 0.06 / under 0.31) |
| Count MAE | 0.47 [0.31, 0.66] |
| Graded nDCG@5 (n=50, bad crops 0, unmatched 0) | 0.676 [0.625, 0.721] |
| P@5 (grade ≥ 2) | 0.652 [0.592, 0.708] |
| Success@1 (top-1 grade ≥ 2) | 0.960 [0.900, 1.000] |
| Mean grade@5 | 1.88 [1.73, 2.03] |
| judged@5 | 0.700 [0.648, 0.748] |
| Negatives rejected | 0.667 [0.333, 1.000] |
| … non_jewellery | 1.000 [1.000, 1.000] |
| … out_of_scope_jewellery | 0.250 [0.000, 0.750] |

| tag | n | count exact | item-category F1 | category acc (first) |
|---|---|---|---|---|
| chain | 4 | 0.000 [0.000, 0.000] | 0.631 [0.500, 0.768] | – |
| overlap | 11 | 0.091 [0.000, 0.273] | 0.667 [0.606, 0.758] | 1.000 [1.000, 1.000] |
| hand_held | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| blur | 1 | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 1.000 [1.000, 1.000] |
| multi_item | 40 | 0.475 [0.325, 0.625] | 0.752 [0.668, 0.824] | 0.941 [0.853, 1.000] |
| no_tag | 22 | 0.909 [0.773, 1.000] | 0.924 [0.818, 1.000] | 0.955 [0.864, 1.000] |

## Phase 8 — summary and integration (2026-10-12)

**Open-set decision** (`src/eval/calibrate.py`): thresholds fixed by a 5% false-reject budget on tune positive photos (negatives are too few to fit on). Holdout reported once.

| holdout | v1 (0.45, no OOS check) | **v2 calibrated** |
|---|---|---|
| False-reject rate, positive photos (n=63) | 0.000 | 0.063 [0.016, 0.127] |
| False-accept rate, negative photos (n=9) | 0.556 [0.222, 0.889] | **0.333** [0.000, 0.667] |
| … non-jewellery (n=5) | 0.20 | **0.00** |
| … out-of-scope jewellery (n=4) | 1.00 | 0.75 |

- Similarity alone cannot separate out-of-scope jewellery: anklet 0.82, ring 0.70, rings 0.69, brooch 0.63 and watch 0.60, against a positive median of 0.71. The calibrated T_low comes out at 0.475, close to v1's 0.45.
- The gain comes from the new **CLIP out-of-scope check** (prompts fixed a priori; one matrix product on the CLIP crop embedding the fused category already computes). The watch is now rejected (p_oos 0.92).
- Its cost: all 4 holdout false rejects are earrings with high similarity (e21, e45, e54, e58; 0.57–0.77). Hoops and studs most likely read as rings to CLIP. Holdout FRR 6.3% is slightly over the 5% budget (tune 5.3%).
- **Closest-matches tier:** score ≥ **0.74**. Over all 173 graded tune results, P(grade ≥ 2) = 0.91 at that point (n=32); holdout check 1.00 (n=64). v1's 0.65 gives 0.75 on tune. Grade 3 ("near-identical") is not predictable from similarity on these photos (tune precision 0.2–0.38 at any threshold), and its use drifted between grading sessions (23% vs 83% of judgements), so the tier means "same design family or better". The margin rule did not improve precision, so MARGIN_THRESHOLD = 0.
- **MMR diversity** (`src/eval/diversity.py`, catalogue val): λ = **0.7**. Clean nDCG@5 0.363 → 0.379, R@5 0.428 → 0.446, ILD@5 0.219 → 0.256, near-duplicate pairs in the top-5 13.9% → 8.4%. Combined-aug nDCG@5 0.183 → 0.185. The diversity gain does not cost relevance; removing cross-shop duplicates lets the right listing into the top 5.
- **Integration:** `config.py` (backup in `artifacts_backup/20261012_121733/`) now has HIGH 0.74 / LOW 0.475 / MARGIN 0.0 / OOS 0.813 / MMR_LAMBDA 0.7, and the duplicate `MARGIN_THRESHOLD` is removed. `decide()` takes `p_oos` and returns a `reason`. `src/pipeline/recommend.py` provides MMR, tiers, the OOS probability and "more like this". `/predict` items now carry `closest_matches`, `similar_items`, `status_reason`, `out_of_scope_probability` and `category_confidence` (`results` kept for the current frontend). New endpoint `GET /similar/<catalogue_id>` (price filter and sort as before; 404 for unknown ids). Live pipeline reproduces the calibration exactly (tune FRR 0.053 / FAR 0.125; holdout 0.063 / 0.333); about 620 ms per photo end to end on CPU.
- Own-photo holdout with MMR: Success@1 0.960 (top-1 unchanged by MMR) and category accuracy 0.946. Graded nDCG@5 awaits the top-up grading: judged@5 0.70; page `data/eval_own/grade_topup_phase8.html` (🙋 optional).

**Phase 8 — MMR on own photos (top-up graded; 7 of 133 left blank and counted as 0, judged@5 ≥ 0.977):**

| own photos | no MMR | MMR λ=0.7 | paired diff [95% CI] |
|---|---|---|---|
| holdout graded nDCG@5 (50 items) | **0.904** | 0.811 | −0.093 [−0.124, −0.060] |
| holdout P@5 | **0.924** | 0.828 | −0.096 [−0.152, −0.044] |
| holdout Success@1 | 0.960 | 0.960 | 0 (MMR keeps the top-1) |
| holdout distinct shops in top-5 | 2.44 | **2.86** | +0.42 [0.24, 0.60] |
| tune graded nDCG@5 (35 items) | **0.634** | 0.595 | −0.039 [−0.069, −0.010] |

On catalogue val, MMR improved relevance by removing cross-shop duplicates of the same product. On real photos, where no exact item exists, the top-5 relevant look-alikes *are* the near-duplicates, and diversifying replaces them with less similar items. Tune agrees with holdout (both CIs exclude 0), so this is not a holdout-only effect.

**Applied (user approved):** `MMR_LAMBDA_PREDICT = 1.0` (MMR off for photo search) and `MMR_LAMBDA_SIMILAR = 0.7` (on for "more like this"). Backup in `artifacts_backup/` (timestamped).

## Phase 9 — frontend (2026-10-12)

Flask + vanilla HTML/JS/CSS, existing pastel style. Backups in `artifacts_backup/20261012_130137/`.
- Each detected item shows the **detected crop** (new `crop_image` in the API, a small JPEG data URL) beside its results, the **fused category with confidence** ("Necklace · 98% confident"), a status chip (Close match / Similar styles / No match) and, for no_match, the reason (out-of-scope text for rings, watches and anklets) with a "show the nearest items anyway" button.
- Results come in two tiers: **Closest matches** (score ≥ 0.74) and **Similar items** (or "Similar styles" when there is no closest match).
- **More like this** on every card opens an in-item panel from `/similar/<id>` (chainable from inside the panel). It respects the price filter and sort.
- Uploaded-photo preview kept. Price filter and sort work for both the photo search and the panel. Catalogue text is HTML-escaped. Responsive at 375 px with no horizontal scroll.
- API fixes made while wiring the UI:
  - (1) `/predict` now calls `preprocess_image()` (EXIF rotation + 1600 px cap; the Phase 0 bug), so crops from rotated phone photos are upright.
  - (2) **Security:** `/catalogue-image/<path>` served any file under the project (`.env` with API keys, `config.py`). It is now restricted to `data/raw/`; `.env`, `config.py` and `../` traversal return 404.
  - (3) `PORT` env var (default 5000); `.claude/launch.json` runs it on 5001 because AirPlay holds 5000.
- Verified in the browser: mixed photo (necklace, 98% confident; relevant oxidised-silver results), out-of-scope watch (no_match + reason + show anyway), exact catalogue photo (1 closest match + 5 similar), chained "More like this", max ₹1500 + price ascending.
- **Data issue found (not fixed):** evryjewels.com prices are in USD (median about 7) but are displayed and filtered as ₹, so a ₹ price filter treats them as very cheap. Fix: per-shop currency conversion in the metadata. Left for the user to decide.

**Currency fix (user approved, 2026-10-12):** Shopify prices are in each store's own currency. `config.SHOP_CURRENCY` (evryjewels.com USD, missoma.com GBP; all other stores INR) and `CURRENCY_TO_INR` (USD 88, GBP 117; fixed, dated rates) are applied at load time in `src/pipeline/filter.load_metadata`; the parquet is unchanged. Price ≤ 0 is treated as unknown (1 listing). Medians in INR after conversion: evryjewels 527 (was "6"), missoma 16,263 (was "139"). Backup in `artifacts_backup/20261012_131149/`.

## Phase 10 — final evaluation (2026-10-12T13:14:15)

Final config: `{'CATEGORY_MODE': 'fused', 'HIGH_MATCH_THRESHOLD': 0.74, 'LOW_MATCH_THRESHOLD': 0.475, 'MARGIN_THRESHOLD': 0.0, 'OOS_THRESHOLD': 0.813, 'MMR_LAMBDA_PREDICT': 1.0, 'MMR_LAMBDA_SIMILAR': 0.7, 'DETECTOR_MODEL': 'google/owlvit-base-patch32', 'EMBEDDER_MODEL': 'facebook/dinov2-base'}`.

**A. Real phone photos (stumper holdout, 73 photos; 95% bootstrap CIs; never used to choose)**

| metric | v1 baseline | v2 final |
|---|---|---|
| Detection recall | 0.984 [0.953, 1.000] | 0.984 [0.953, 1.000] |
| Category accuracy (first item) | 0.589 [0.464, 0.732] | 0.946 [0.893, 1.000] |
| Item-category F1 | 0.517 [0.413, 0.619] | 0.813 [0.745, 0.872] |
| Count exact | 0.629 [0.484, 0.758] | 0.629 [0.484, 0.758] |
| Graded nDCG@5 | 0.899 [0.841, 0.948] | 0.904 [0.837, 0.955] |
| P@5 (grade ≥ 2) | 0.924 [0.864, 0.980] | 0.924 [0.856, 0.980] |
| Success@1 | 0.880 [0.780, 0.960] | 0.960 [0.900, 1.000] |
| judged@5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| False-reject rate (positive photos) | 0.000 [0.000, 0.000] | 0.063 [0.016, 0.127] |
| False-accept rate (negative photos) | 0.556 [0.222, 0.889] | 0.333 [0.000, 0.667] |
| … non-jewellery | 0.200 [0.000, 0.600] | 0.000 [0.000, 0.000] |
| … out-of-scope jewellery | 1.000 [1.000, 1.000] | 0.750 [0.250, 1.000] |

**B. Catalogue retrieval, test split (same-listing protocol)**

| metric | v1 baseline (legacy re-rank, HNSW) | v2 final (fused re-rank) |
|---|---|---|
| R@1 | 0.314 | 0.318 |
| R@5 | 0.438 | 0.440 |
| R@10 | 0.497 | 0.497 |
| MRR | 0.376 | 0.379 |
| nDCG@5 | 0.383 | 0.386 |
| category P@5 (non-set) | 0.957 | 0.971 |
| graded nDCG@5 (trusted labels) | – | 0.730 |

Per augmentation (test, R@1 / MRR):

| aug | v1 baseline | v2 final |
|---|---|---|
| clean | 0.320 / 0.376 | 0.318 / 0.379 |
| blur | 0.133 / 0.183 | 0.133 / 0.185 |
| low_light | 0.300 / 0.355 | 0.300 / 0.355 |
| bright_contrast | 0.280 / 0.343 | 0.280 / 0.344 |
| rotate | 0.280 / 0.341 | 0.280 / 0.343 |
| crop_zoom | 0.277 / 0.341 | 0.280 / 0.345 |
| jpeg | 0.313 / 0.365 | 0.313 / 0.366 |
| busy_bg | 0.083 / 0.129 | 0.087 / 0.134 |
| perspective | 0.303 / 0.360 | 0.303 / 0.361 |
| combined | 0.137 / 0.186 | 0.140 / 0.190 |

**C. Latency per stage, final pipeline (CPU, one image at a time, per detected item unless noted)**

| stage | p50 ms | p95 ms | n |
|---|---|---|---|
| preprocess | 13.5 | 22.9 | 142 |
| detect | 261.1 | 272.6 | 142 |
| crop | 0.2 | 0.5 | 196 |
| embed (DINOv2) | 95.0 | 103.1 | 196 |
| search (HNSW) | 0.2 | 0.3 | 196 |
| category (kNN + CLIP + clf) | 33.5 | 36.9 | 196 |
| out-of-scope check | 0.0 | 0.0 | 196 |
| re-rank | 0.0 | 0.0 | 196 |
| decide | 0.0 | 0.0 | 196 |
| MMR + tiers | 0.0 | 0.0 | 196 |
| TOTAL per photo | 410.1 | 681.8 | 142 |

## Phase 10 — ablation by phase (what each change contributed; numbers from each phase's own evaluation)

| phase | change | adopted? | key evidence |
|---|---|---|---|
| 1 | Evaluation harness (same-listing protocol, 8 phone augmentations, pHash variant, CIs) | yes (eval) | First valid retrieval baseline: test R@1 0.314, MRR 0.376. Found the classifier input bug (live classifier = constant "earring"). |
| 2 | Metadata backfill (variant map → title rules → CLIP+DINOv2 probe) | yes (eval/metadata) | Subtype blanks 60% → 18%, colour 87% → 14%; user spot-check 95% / 96% correct. Trusted-label metrics guard against model-label bias. |
| 3 | Own-photo labels + graded set | yes (eval) | 140 photos labelled, 50 holdout items graded. Showed undercounting (31%) and that the UI's detector label (0.857) beats the classifier (0.589). |
| 4 | Fused category (kNN + CLIP + retrained classifier) | **yes** | Holdout category acc 0.589 → **0.946**; necklace→earring errors 14 → 1; Success@1 0.88 → **0.96**; +28 ms/item. |
| 5 | Detector benchmark (OWLv2, Grounding DINO), prompt ensembles, NMS rules, SAHI tiles, verification | no | No change beat v1 by the 0.02 F1 margin on tune. 20/25 undercounts are recall failures (small items inside a necklace box), not merge errors. Tune-best config was worse on holdout (count 0.629 → 0.548). Grounding DINO kept in reserve (+0.024 F1, 16× slower). |
| 6 | Embedding variants (background removal, CLIP, DINOv2+CLIP, listing aggregation) | no | DINOv2 alone best (val primary 0.264). Background removal helps busy backgrounds only (+0.03 MRR) and hurts blur (−0.12). Mean aggregation far worse (R@1 0.076). |
| 7 | Contrastive projection heads (listing positives; augmentation-only positives) | no | Catalogue test R@1 0.316 → 0.434 (FT), but own-photo holdout nDCG@5 0.90 → 0.64 (44/50 items worse); FTaug also worse on tune (−0.096). Both double the wrong-category share of the top-5. Key lesson: same-listing Recall@k ≠ similar-item quality. |
| 8 | Calibrated decision + CLIP out-of-scope check | **yes** | Holdout false-accept 0.556 → **0.333** (non-jewellery 0.20 → 0.00), false-reject 0 → 0.063. Closest tier at 0.74 (P(grade ≥ 2) 0.91 tune / 1.00 holdout). |
| 8 | MMR diversity | photo search: **no**; "more like this": yes | Catalogue val nDCG@5 +0.016, but own-photo holdout nDCG@5 −0.093 (tune −0.039). |
| 9 | Frontend (tiers, crop, category + confidence, more-like-this) + API fixes | **yes** | EXIF preprocessing in `/predict`; `/catalogue-image` path hardening (`.env` was readable); currency conversion (USD/GBP stores). |

Final vs v1 on holdout: category accuracy +0.357, item-category F1 +0.296, Success@1 +0.08, graded nDCG@5 +0.005 (n.s.), false-accept −0.22, false-reject +0.063. Counting is unchanged (0.629).

**Phase 10 docs (2026-10-12):** `writeup.md` created (v1 Parts A/B reconstructed from `writeup.pdf` with marked v2 corrections; new Part C with methodology, final results, what worked and what did not, remaining failures, caveats). v2 sections appended to `decisions.md` and `README.md`; no existing text was removed. 🙋 Outstanding: about 5 UI screenshots per category for the writeup.
