# DECISIONS.md — Jewel Lens (Stump the Model, Problem 2)

## Category choice

We chose jewellery — specifically earrings, necklaces, and bracelets — over watches, eyewear, or footwear. Earrings and necklaces in particular are common, everyday-worn items with huge design variety (colour, material, subtype), which gives a catalogue-matching system real versatility to demonstrate on. This category also mirrors how Dyla itself started out, making it a natural, closer-to-home choice to build against.

## Architecture

```mermaid
flowchart TD
    A[User photo] --> B[Preprocess<br/>EXIF fix, resize, RGB]
    B --> C[OWL-ViT<br/>zero-shot open-vocab detection]
    C -->|0 boxes| D[No jewelry detected]
    C -->|N boxes| E[Crop each box<br/>+12% padding]
    E --> F[DINOv2<br/>frozen embedding, per crop]
    F --> G[Category classifier<br/>logistic regression on embedding]
    F --> H[FAISS HNSW<br/>single flat index, top-20]
    G -->|soft boost| H
    H --> I[Decision<br/>similarity + margin threshold]
    I --> J[MATCH / UNCERTAIN / NO MATCH]
    J --> K[Price/relevance filter]
    K --> L[Top-5 + confidence, per item]

    subgraph Offline
    M[Scrape: Shopify /products.json<br/>3 categories x subtype x colour] --> N[Perceptual-hash dedup]
    N --> O[CLIP zero-shot label QA]
    O --> P[DINOv2 embed catalogue]
    P --> Q[Build FAISS index]
    end
```

**Pipeline:** photo → OWL-ViT detection (zero-shot, prompted per category) → per-box crop → DINOv2 embedding (frozen) → category classifier (soft re-rank signal only) + FAISS similarity search (single flat HNSW index) → similarity+margin decision → price/relevance filter → top-5 with confidence, independently per detected item.

**Catalogue:** ~3,800 listings / ~13,300 images, scraped from public Shopify `/products.json` endpoints across multiple Indian and Western jewellery stores, spanning 3 categories (earring, necklace, bracelet — jhumka etc. tracked as subtype metadata, not a separate class) and a 16-colour palette.

## All decisions made

| Component | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Detection model | OWL-ViT (zero-shot open-vocab) | YOLO (fine-tuned) | No bounding-box training data available; COCO has no jewellery classes. Zero-shot text-prompted detection needed no labelled boxes at all. |
| Detection post-processing | NMS added | None (initial build) | Duplicate/overlapping boxes on the same object were a confirmed real bug in Phase 3 testing, not a labelling issue. |
| Detection prompts | Moderately descriptive phrase per category, tuned across 3 rounds | Bare category word; heavily elaborate descriptive phrase | Bare words under-specified; elaborate wording sometimes regressed correct detections (hurt jhumka recall specifically). Settled on the middle version based on measured results, not assumption. |
| Embedding model | DINOv2 (frozen) | CLIP | DINOv2 favoured for fine-grained visual discrimination (near-identical design variants) over CLIP's semantic/text-aligned space. CLIP still used separately for catalogue label QA, where zero-shot semantic classification is the actual need. |
| Retrieval index | FAISS HNSW, single flat index | Category-partitioned HNSW; IVF-PQ; brute-force | Partitioning solves a latency problem we don't have at ~13k items, and risks permanently excluding the correct item if the detector's unreliable category label routes to the wrong sub-index (confirmed happening in testing). IVF-PQ's memory savings only matter past ~1M items. Brute-force was considered as an honest baseline — comparably fast at this scale, but no headroom if the catalogue grows. |
| Category classifier | Logistic regression (C=0.01, `class_weight="balanced"`) | SVM; Naive Bayes | Logistic regression gives calibrated probabilities natively, needed for the soft re-rank/confidence signal, without SVM's extra Platt-scaling step. Naive Bayes rejected for violating the feature-independence assumption on correlated embedding dimensions. |
| Classifier regularization | C swept via 3-fold CV, C=0.01 selected | Default C=1.0 | Default showed a train/val gap (100% vs 96.66%) indicating mild overfit. The sweep improved *both* held-out splits, not just closing the gap — confirmed generalization improvement, not just regularization for its own sake. |
| Classifier's role in retrieval | Soft re-rank boost on top-20 results | Hard category routing / pre-filtering | Routing risks permanently losing the correct item when the classifier or detector mislabels category — confirmed happening in real testing. A soft boost keeps retrieval similarity as the dominant, better-generalizing signal. |
| Confidence / no-match decision | Similarity + margin thresholds, set from a small manual sample | Full ROC/FAR-FRR calibration | Descoped in favour of doing the multi-item extension properly rather than two shallow extensions, per the brief's own guidance. |
| Catalogue source | Public Shopify `/products.json` scraping | Etsy API; eBay API; stock-photo APIs (Pexels/Pixabay); purchased dataset (DataSeeds.ai) | Etsy access was rejected/banned; eBay approval didn't arrive in time. Stock-photo APIs have no real per-item structure or price and thin jewellery-specific volume. A purchased dataset would conflict with the brief's "assemble the catalogue yourself by scraping" requirement. |
| Category taxonomy | 3 top-level classes (earring/necklace/bracelet); jhumka etc. tracked as subtype metadata | Jhumka as a 4th top-level class | Jhumka is genuinely a subtype of earring, not a sibling category. Keeping it as metadata avoids taxonomy ambiguity in classifier training and in the confused-pairs analysis. |
| Colour palette | 16 colours (metallic + stone/finish), applied uniformly across categories | Gold/silver only; custom per-category lists | An initial 2-colour palette was too thin to satisfy "different colours" coverage. A uniform full palette gave consistent facet coverage without extra scraping logic per category. |
| Catalogue cleaning | Two-stage: perceptual hashing (pHash) then CLIP zero-shot label QA | Single-stage exact-hash dedup only; no dedup | pHash catches near-identical/resized stock photos reused across different sellers — exact-hash matching misses resized/recompressed duplicates. CLIP QA separately catches category-mismatched images that scraping's keyword filter still let through — a different failure mode from duplication, needing a different check. |
| Multi-item handling | Natural consequence of running the pipeline once per detected box | A separate multi-item detection/matching model | The architecture already produces independent crops per box; no additional model was needed. Verified directly via spatial-distinctness and duplicate-result checks (7/7 multi-item test photos passed cleanly). |
| Price/relevance filter | Post-retrieval metadata filter/sort | A ranking model incorporating price as a learned signal | Kept as pure post-processing on already-computed results — no retraining, no new signal, deliberately capped effort as polish rather than core work. |
| Shop name, price, and listing URL shown per result | Displayed alongside every returned candidate | Returning only an internal catalogue ID/score | A bare similarity score doesn't tell a user anything actionable. Since every catalogue item traces back to a real store listing, surfacing which shop it's from and its price gives people an immediate, concrete sense of the match (a real product they could actually go look at or buy), rather than an abstract "confidence: 0.82" — it's what turns a retrieval result into something a person can actually use. |
| Frontend stack | Flask + vanilla HTML/JS | Streamlit; React | Streamlit is fastest to build but has no resume value, which mattered for this submission. Flask + vanilla JS balances resume relevance against build-time cost without a frontend framework's setup overhead. |
| Scrape reliability | Per-domain progress tracking with retry/backoff, resumable across runs | One-shot script, no resume | Etsy's rate limits (5 QPS / 5000 QPD) and Shopify storefronts' own limits made crash-safe resumability necessary, not optional, after an early run was interrupted mid-scrape. |
| Learned matching model | Frozen pretrained DINOv2, no fine-tuning by default | CNN trained from scratch; Siamese/triplet network; cross-encoder reranker | All three need paired training data (same physical item, multiple real photos) we didn't reliably have going in. Cross-encoder also multiplies inference cost per query (one forward pass per candidate). Not worth the build risk in 3 days — flagged as a "with more time" item instead. |
| LLM/VLM usage | Offline only — catalogue label QA and stumper-photo labelling assist | An LLM/VLM call inside the live matching pipeline | An online VLM call adds hundreds of ms to seconds per query, directly undermining the latency requirement, and isn't more accurate here than our calibrated embedding-similarity signal, which we can inspect and explain — a VLM's yes/no can't be interrogated the same way. |


## Trade-offs under the time limit

- **Category classifier has a documented bias**: trained with `class_weight="balanced"`, C-tuned via CV (97% test accuracy), but its intercepts are skewed toward the majority class (earring), and it defaults to "earring" on several real (non-catalogue-domain) test photos even when clearly wrong. We mitigated this by keeping its re-rank influence deliberately small (`RERANK_BOOST_WEIGHT=0.08`) so retrieval similarity — the stronger, better-generalizing signal — dominates final ranking. Confirmed on catalogue-style test images that even in-distribution photos triggered the same bias, ruling out "just a domain-gap problem."
- **Thresholds (similarity/margin) are hand-set from a small manual sample** (18 positives, 6 negatives), not a full ROC sweep — acceptable given the descoped-open-set decision, but not statistically robust.
- **Subtype and colour metadata are ~60% blank** — title-keyword extraction from scraped Shopify data is imperfect; planned VLM backfill wasn't completed.
- **A handful of self-owned physical items were added to the catalogue** (studio-style reference photo) specifically so Part B could shoot verified hard-condition photos of items with known ground truth, since most scraped catalogue items aren't physically accessible to us.

## Known failure modes (diagnosed, not guessed)

- **Round/circular-shaped earrings (jhumka, hoops) are frequently mislabelled `bracelet` by OWL-ViT**, worse when held in a hand — confirmed across three independent prompt variants; this is a genuine zero-shot detector limitation, not a bug.
- **Retrieval (DINOv2+FAISS) is more robust than either the detector's or classifier's category label** — in every case checked, the actual returned images were visually/stylistically correct even when the upstream label was wrong.
- **Multi-item detection is structurally sound**: 7/7 multi-item test photos passed spatial-distinctness and independent-retrieval checks, with zero duplicate-crop or duplicate-result bugs. A 4-item photo does compound detector-labelling risk (4 chances to mislabel vs. 1), which is a separate, real limitation from the multi-item mechanism itself.
- **Detection is the dominant latency cost**, not FAISS retrieval (confirmed via per-stage profiling) — retrieval is near-instant regardless of catalogue growth at this scale.

## Testing

**Quantitative results** (stumper set: 183 non-negative photos across earring/necklace/bracelet/mixed folders, 12 negatives; ground truth from folder structure, independent of the retrieval/classifier models being tested):

| Metric | Result |
|---|---|
| Detection Recall | 1.000 (183/183 jewelry photos got ≥1 box) |
| Category Accuracy | 0.665 |
| No-match precision (negatives correctly rejected) | 0.583 |
| Confused pairs | necklace → earring: 28, bracelet → earring: 27 |

Exact-match retrieval metrics (Recall@1/5, MRR) were descoped: the stumper photos are the author's own personal jewelry, never listed on any scraped store, so there is no catalogue-ID ground truth to match against — the "correct" item may legitimately not exist in the catalogue at all. Retrieval quality was instead evaluated qualitatively (below), via a full visual review of every stumper photo's top-5 results (`review_report.html`).

**Positives found in review:**
- Detection boxes are accurate on the large majority of clean, single-item photos.
- Blurred photos were still handled well — DINOv2 embeddings are reasonably blur-robust.
- Multi-item photos mostly detect and retrieve each item independently and correctly.
- Even when the upstream category label was wrong, the retrieved images were still visually/stylistically close to the query (consistent with the "retrieval is more robust than the label" finding above).

**Failures found in review, beyond what training-time testing had already caught:**
- **Chain necklaces (4 photos) get cropped "2 and 2"** — the detector proposes two boxes near the chain's two ends, and since they barely overlap, NMS (which only suppresses high-IoU pairs) treats them as two separate objects instead of merging them into one continuous chain.
- **Overlapping earrings (worn as a pair) sometimes collapse into a single detection** — the mirror-image failure of the chain issue: two genuinely distinct, overlapping items produce high-IoU boxes, so NMS discards one as a "duplicate." Same root cause (NMS is a purely geometric heuristic, blind to object identity) producing opposite symptoms.
- **Necklace pendants get detected/classified as earrings** — a solitary pendant, cropped in isolation, has the same small/dangling silhouette OWL-ViT associates with "earring." Same shape-based root cause already documented for jhumka/hoop → bracelet, just in the other direction.
- **Bracelets held in the hand get cropped too small and misclassified as earring** — hand occlusion reduces the detection box to a small curved segment (a clasp or charm) rather than the full bracelet, and that reduced crop reads as earring-shaped.
- **Mixed-folder photos where 1 item is detected but the result retrieved is a multi-piece "set" listing** — several catalogue listings are themselves sets (e.g. "necklace and earring set") but carry a single category label from keyword matching; a single-item crop can legitimately have a set-listing as its nearest visual neighbour. This is a catalogue-labelling limitation, not a detection or retrieval bug.

## With two more weeks

1. Fix the classifier's intercept bias properly (class-balanced resampling instead of relying on `class_weight`) and re-validate against real photos, not just held-out catalogue data.
2. Run a proper ROC-calibrated open-set threshold sweep using a much larger negative set.
3. Backfill subtype/colour metadata via VLM-assisted QA on the ~60% currently blank.
4. Investigate a lightweight contrastive fine-tune of DINOv2's last layers using the real same-item photo pairs captured from multi-image Shopify listings — a more honest training signal than synthetic augmentation alone.
5. Automate the stumper set (programmatic hard-case generation) and compare its defeat rate against the hand-shot set.
6. Scale-test the catalogue toward 100k items and re-profile latency under that load with `efSearch` tuning.
7. **Expand filtering beyond price** — the current filter mirrors what every shopping app offers (sort/filter on results), but we scoped it to price only since it's the single most-used filter for a first version. With more time this extends naturally to material, colour, and subtype filters using metadata we already scrape but haven't fully backfilled (point 3 above unblocks this).
8. A "find similar to this result" loop — clicking a returned catalogue item re-queries the index using that item's own embedding, letting a user browse outward from a match instead of only searching from their own photo.
9. Lightweight feedback capture (thumbs up/down per result) to build a real labelled dataset over time for retraining the classifier and calibrating thresholds, rather than relying on our small hand-built test sets.
10. Revisit the dress→jewellery cross-domain recommendation idea we deliberately cut early — a genuinely different, higher-effort feature, better suited to a longer build.
11. Merge same-class adjacent boxes for elongated objects (chains) and reconsider NMS's overlap-only merge logic so it stops conflating "two segments of one object" with "two distinct overlapping objects" — the two review-set failures documented above are opposite symptoms of that same fix.
---

# v2 decisions (2026-10, Phases 0–10)

Each entry lists what was decided, the alternatives tested and the measured evidence. "Tune" and "holdout" are the two halves of the stumper set, split by photo and stratified by folder; choices were made on tune (or catalogue val) and reported on holdout. Full tables are in `results/LOG.md`.

## Evaluation design
- **Decision:** two complementary benchmarks.
  - The **same-listing protocol** on a fixed listing-level catalogue split (query = one image of a listing; relevant = the same listing; query image excluded).
  - **Graded own-photo evaluation** (0–3 grades on real phone photos).
- **Alternative rejected:** v1's CLIP auto-labelling of stumper ground truth. It is circular, because it picked the "correct" item from inside the top-5.
- **Evidence it mattered:** Phase 7 improved same-listing test R@1 from 0.316 to 0.434 while lowering own-photo graded nDCG@5 from 0.90 to 0.64. Either benchmark alone would have given the wrong decision.
- **Anti-overfitting rules:** tune/holdout split by folder; bootstrap CIs; a margin rule (a change must beat the current choice by more than 0.02 tune F1). Without that rule, Phase 5's tune-best detector config lowered holdout count accuracy from 0.629 to 0.548.

## Category classifier: the root cause of the earring bias
- **Finding:** the classifier was trained on raw DINOv2 CLS vectors (norm ≈ 46) and fed L2-normalised ones (norm 1). In production it predicted "earring" for 13,352/13,352 catalogue images. The intercepts were a symptom, not the cause.
- **Decision:** a fused category (`src/pipeline/category.py`, `CATEGORY_MODE="fused"`), the equal-weight average of:
  - a kNN vote over retrieved neighbours;
  - CLIP zero-shot on the crop (prompts chosen on train catalogue images);
  - a classifier retrained on normalised clean, augmented and detector-crop views (balanced weights, temperature-scaled, val ECE 0.007).
- **Alternatives tested:** each signal alone, every equal-weight subset, a weighted grid, and logistic-regression stacking (leave-one-photo-out tune accuracy 0.773, not adopted). The detector label gets weight 0.
- **Evidence:** holdout category accuracy 0.589 → 0.946; item-category F1 0.517 → 0.813; Success@1 0.88 → 0.96. Cost: +28 ms per item.

## Metadata backfill
- **Decision:**
  1. Map the 33 Shopify colour strings to the palette.
  2. Apply category-aware title rules.
  3. Fill remaining blanks with a linear probe on CLIP+DINOv2 listing features, trained on train-split known labels, with a calibrated threshold and per-class gating.
- **Alternative rejected:** CLIP zero-shot (39% subtype top-1 accuracy; 3% coverage at usable precision).
- **Evidence:** user spot-check 42/44 subtype and 53/55 colour correct.
- **Guard:** the "trusted" metrics use only title/variant labels, so image-model labels cannot flatter image-model retrievers.

## Detection (no change adopted)
- **Tested on tune:** OWLv2, Grounding DINO-tiny, prompt ensembles (max/mean, plus generic "jewellery"), class-agnostic vs per-class NMS, pair protection, group-box drop, chain merge, minimum box size, SAHI-style tiling, retrieval verification.
- **Evidence:**
  - Grounding DINO: best tune F1 (0.732 vs 0.708) but about 16× slower and below the margin. Kept in reserve.
  - Diagnosis: 20 of 25 undercounted photos have no box above threshold even before NMS. The real failure is recall on small items inside larger ones, which the NMS rules cannot fix.
  - Tiling cut undercounting (0.41 → 0.34) but added overcounting (0.11 → 0.20).
- **YOLO on pseudo-labels:** skipped (user decision). Pseudo-labels from the same detector would inherit its misses.

## Embedding (no change adopted)
- **Tested on val:** DINOv2 alone, DINOv2 on background-removed images (rembg/u2net), CLIP B/32, DINOv2+CLIP score fusion (weight grid), and listing scoring by mean / max+mean of images.
- **Evidence (primary = mean MRR of clean and combined-aug queries):**

  | variant | primary |
  |---|---|
  | DINOv2 | **0.264** |
  | background removed | 0.233 |
  | CLIP | 0.163 |
  | best fusion | 0.263 |
  | mean aggregation | 0.090 |

  Background removal helps busy backgrounds (MRR +0.028) but hurts blur (−0.118) and is 3.6× slower.

## Contrastive fine-tuning (not adopted)
- **Tested:** a linear or residual-MLP projection head on frozen DINOv2 (identity initialised, InfoNCE, in-batch and hard negatives, early stopping on val), with listing positives and, separately, augmentation-only positives.
- **Evidence:**
  - Listing-positive head: test R@1 0.316 → 0.434, but own-photo holdout graded nDCG@5 0.90 → 0.64 (paired CI −0.33 to −0.19); wrong-category share of top-5 8% → 20%.
  - Augmentation-only head: tune nDCG@5 0.635 → 0.538, failing the rule set in advance, so it never reached holdout.
- **Reason:** instance discrimination on catalogue photos learns shop or listing style rather than design similarity.

## Open-set decision and tiers
- **Decision:**
  - LOW 0.475 (2.5% false-reject budget on tune positives);
  - a CLIP out-of-scope check (fixed prompts, threshold 0.813, the other 2.5%);
  - HIGH 0.74 = "Closest matches" (P(grade ≥ 2) ≥ 0.9 on tune);
  - margin 0, because it did not improve precision.
- **Why not fit on negatives:** there were only 8 tune negatives, so thresholds control the false-reject rate on positives instead.
- **Why grade ≥ 2 for the tier:** grade 3 was not predictable from similarity and drifted between grading sessions.
- **Evidence (holdout):** false-accept 0.556 → 0.333 (non-jewellery 0.20 → 0.00); false-reject 0 → 0.063, all of them earrings read as rings.

## Diversity (MMR)
- **Decision:** λ = 0.7 for "more like this"; off for photo search.
- **Evidence:** catalogue val nDCG@5 +0.016 and near-duplicate pairs 13.9% → 8.4%. Own-photo holdout nDCG@5 −0.093 (tune −0.039), because real-photo look-alikes are often near-duplicates of each other.

## API / data fixes
- `/predict` now preprocesses uploads (EXIF rotation).
- `/catalogue-image` is restricted to `data/raw/`; it previously served `.env` and `config.py`.
- Prices are converted to INR per shop (evryjewels USD, missoma GBP).
- The API port is configurable through `PORT`.

## Status of v1's "With two more weeks" list
1. Classifier bias: fixed (it was the input bug; see above).
2. Calibrated open-set threshold: done, but the negative set is still small (9 holdout).
3. Metadata backfill: done (probe, not a VLM).
4. Contrastive fine-tune: done as a head on frozen features. It helps catalogue metrics and hurts real-photo recommendations, so it was not adopted.
8. "Find similar to this result": done (`/similar/<id>`, "More like this").
11. Chain merge and NMS logic: implemented and ablated. It did not help, because those failures are mostly missing boxes, not merges.
- Not done: 5 (automated stumper generation), 6 (100k-scale latency), 7 (colour and subtype filters in the UI; the metadata now supports them), 9 (feedback capture), 10 (dress → jewellery).
