# Jewel Lens — Stump the Model, Problem 2

Image-based jewellery recommender in the spirit of Google Lens: upload a photo, and for every detected earring, necklace or bracelet get the closest catalogue pieces (shop, price, link), plus "more like this" browsing. Design rationale and every decision with its evidence are in `decisions.md`; measured numbers for every phase are in `results/LOG.md`.

> Parts A and B below are the v1 submission, reconstructed from `writeup.pdf`. Its screenshots are not reproduced here, only marked by placeholders. Where v2's evaluation showed a v1 statement to be wrong, the statement is kept and a **v2 correction** note follows it. Part C (new) is the v2 work.

---

## Part A — Catalogue Matcher (v1)

**Catalogue:** ~3,800 listings / ~13,300 images, scraped from public Shopify `/products.json` endpoints across 8 stores, spanning 3 categories (earring, necklace, bracelet) and a 16-colour palette, cleaned via perceptual-hash dedup + CLIP zero-shot label QA.

> **v2 correction:** the cleaned catalogue has **3,513 listings** and 13,352 indexed images from **7 stores**. Prices were in three currencies (INR, USD for evryjewels.com, GBP for missoma.com). v2 converts them to INR (Part C).

**Pipeline:** photo → EXIF/resize preprocessing → OWL-ViT zero-shot detection (+ NMS) → per-item crop → DINOv2 frozen embedding → FAISS HNSW similarity search (top-20) + logistic-regression category classifier (soft re-rank boost only) → similarity + margin decision (match / uncertain / no-match) → price/shop metadata attached → top-5 returned, independently per detected item.

> **v2 correction:** the API never called the preprocessing step, so EXIF rotation was not applied to uploads (fixed in v2). The classifier was trained on raw DINOv2 vectors but fed L2-normalised ones, so in production it predicted "earring" for every input (fixed in v2).

**No-match handling:** a query is only called a match when top-1 similarity clears a high threshold and has a clear margin over the runner-up; otherwise it's uncertain or no_match. Measured no-match precision on held-out negatives: 0.583.

> **v2 correction:** that figure is the share of negative photos rejected (a true-negative rate), not a precision. On the fixed v2 holdout half it was 0.444 for v1.

**Lookup speed:** FAISS retrieval is near-instant at this catalogue size; detection is the dominant per-query cost. *(v1 left a placeholder for measured latency; v2 measured it, see Part C.)*

**Multi-item extension:** each detected box runs independently through the full pipeline. Verified via spatial-distinctness and duplicate-result checks: 7/7 multi-item test photos passed.

> **v2 correction:** with per-photo item counts labelled by hand (Phase 3), the detector's count is correct on 63% of holdout photos. It mostly undercounts (31%), typically missing earrings that lie inside a necklace's loop.

**Frontend** — *[screenshot: upload screen, writeup.pdf p.1]*

**Sample outputs — Part A** — *[screenshots, writeup.pdf pp.2–7: earring (pair of hoops, two items), necklace (two examples), bracelet (cuff), and a non-jewellery photo (toy figurines) correctly rejected with no detections]*

---

## Part B — Stumper Test Set (v1)

**Construction:** 100+ self-shot hard photos, organised by folder (earring/, necklace/, bracelet/, mixed/, negatives/), deliberately covering bad lighting, awkward angles, occlusion (hand-held items), blur, reflective surfaces and multi-item frames. negatives/ holds real non-jewellery objects to test rejection.

**Harness & metrics** — ground truth from folder structure:

| Metric | Result (v1) |
|---|---|
| Detection Recall | 1.000 (183/183 jewellery photos got ≥1 detection) |
| Category Accuracy | 0.665 |
| No-match precision (negatives correctly rejected) | 0.583 |
| Confused pairs | necklace → earring: 28, bracelet → earring: 27 |

> **v2 correction:** there are 136 photos (125 positive) plus 6 extra negatives. "183" counted labelled item rows. The 0.665 category accuracy is the share of earrings in the set, because the live classifier always answered "earring". The confusions (28, 27) are every necklace and bracelet item.

Exact-match retrieval metrics were not reported for the stumper photos because they are the author's own jewellery and have no catalogue ground truth. Retrieval quality was judged qualitatively in `review_report.html`.

> **v2:** retrieval is now measured two ways: the same-listing protocol on the catalogue (Recall@k, MRR, nDCG) and graded relevance on your own photos (Part C).

**Error analysis, positives (v1):** detection boxes are accurate on most clean single-item photos; blurred photos were handled reasonably; multi-item photos mostly retrieve each item independently; even with a wrong category label, retrieved items were visually close.

**Error analysis, failures (v1):** classifier bias toward earring; round items (jhumka, hoops) labelled bracelet by OWL-ViT; pendants labelled earring; hand-held bracelets cropped small; NMS splitting chains and merging overlapping earrings; single items retrieving multi-piece "set" listings.

> **v2:** the classifier "bias" was the input-normalisation bug, not class imbalance. The NMS diagnosis was mostly wrong for this photo set: 20 of 25 undercounted tune photos never get a box at all, so they are recall failures, not merge failures (Phase 5).

---

## Part C — v2: making it measurable, then improving it

### C1. How it was evaluated

- **Catalogue retrieval (same-listing protocol).**
  - Each eligible listing in a fixed listing-level split (70/15/15, stratified by category, seed 42) gives one query image. The relevant item is the same listing, and the query image itself is excluded from the index.
  - Reported: R@1/5/10, MRR, nDCG@5, category P@5, and a graded nDCG@5 using subtype labels taken only from titles and variants, so labels predicted by an image model cannot flatter it.
  - The same queries are also run through 8 phone-photo augmentations (blur, low light, contrast, rotation, crop, JPEG, busy background, perspective) plus a combined one.
  - Chosen on val; test evaluated once.
- **Real phone photos (stumper set).**
  - Split 50/50 by photo, stratified by folder: *tune* for choosing settings, *holdout* for reporting.
  - The author labelled per-photo item counts and failure tags (140 photos) and graded the top-5 results 0–3 for 50 holdout items, with later top-up and pooled rounds.
  - Graded nDCG@5 uses a fixed ideal (grade 3 at every rank), so different systems are comparable.
  - Bootstrap 95% CIs throughout.
- **Rules** (enforced by the scripts):
  - No stumper photo is ever used for training.
  - Test and holdout are never used to choose.
  - Any adopted change must beat the current choice by a fixed margin on tune.

### C2. Final results (v1 baseline vs v2 final)

**Real phone photos — holdout (73 photos, never used to choose):**

| metric | v1 | v2 final |
|---|---|---|
| Category accuracy (first item) | 0.589 [0.46, 0.73] | **0.946** [0.89, 1.00] |
| Item-category F1 (vs labelled counts) | 0.517 | **0.813** |
| Success@1 (top result same design family or better) | 0.880 | **0.960** |
| Graded nDCG@5 | 0.899 | 0.904 |
| P@5 (grade ≥ 2) | 0.924 | 0.924 |
| Negative photos wrongly accepted | 0.556 | **0.333** |
| … non-jewellery objects | 0.20 | **0.00** |
| … rings / watch / anklet / brooch | 1.00 | 0.75 |
| Positive photos wrongly rejected | 0.000 | 0.063 |
| Count exact | 0.629 | 0.629 |
| Detection recall | 0.984 | 0.984 |

**Catalogue retrieval — test split (493 queries):** R@1 0.314 → 0.318, MRR 0.376 → 0.379, category P@5 0.957 → 0.971. The embedding is unchanged, so retrieval itself barely moves. Every phone-photo augmentation is equal or slightly better.

**Latency (CPU, final pipeline):** about 410 ms per photo at p50 (680 ms p95). Detection takes 261 ms; per item, DINOv2 embedding 95 ms, fused category 34 ms (mostly CLIP), and HNSW search 0.2 ms. The out-of-scope check, re-rank and decision are each under 0.1 ms.

### C3. What worked

1. **Fused category identification (Phase 4).** Root cause first: the classifier's input normalisation. The final category is an equal-weight average of three signals, chosen on tune from five candidates and their combinations:
   - a kNN vote over the retrieved catalogue neighbours;
   - CLIP zero-shot on the crop (prompts chosen offline on catalogue images);
   - a classifier retrained on clean, augmented and detector-crop views (ECE 0.007 on val).

   The detector label is kept as an input but gets no weight. Holdout category accuracy rose from 0.589 to 0.946, and necklace→earring errors fell from 14 to 1.
2. **Calibrated open-set decision (Phase 8).**
   - Similarity alone cannot reject out-of-scope jewellery: an anklet scores 0.82 against a median of 0.71 for real pieces.
   - A CLIP out-of-scope check (fixed prompts such as "a ring", "a watch", "an anklet"), calibrated to a 2.5% false-reject budget on tune positives, rejects every non-jewellery negative and some rings and watches.
   - The "Closest matches" tier (score ≥ 0.74) has 91% same-design-family precision on tune and 100% on holdout.
3. **Recommender features (Phases 8–9).** Two result tiers, "More like this" (`/similar/<id>`, with MMR diversity λ = 0.7, which raised val nDCG@5 by 0.016), the detected crop and category confidence in the UI, and per-photo explanations for "no match".
4. **Metadata backfill (Phase 2).** Blank subtypes fell from 60% to 18% and blank colours from 87% to 14%, with 95–96% correct in a user spot-check.

### C4. What did not work (and why that matters)

- **Detection (Phase 5).**
  - Nothing beat OWL-ViT's v1 configuration by the margin. I tried OWLv2, Grounding DINO, prompt ensembles, NMS variants, SAHI-style tiles and retrieval verification.
  - Grounding DINO was best on tune (+0.024 F1) but 16× slower; it is kept in reserve.
  - A tune-optimised configuration *without* the margin rule made holdout worse (count exact 0.629 → 0.548), which is why the margin rule exists.
- **Embedding variants (Phase 6).**
  - Background removal (rembg/u2net) helps busy backgrounds but breaks blurred photos (MRR 0.180 → 0.062).
  - CLIP is much less robust to phone conditions than DINOv2, and fusing the two never beats DINOv2 alone.
- **Contrastive fine-tuning (Phase 7)** — the main negative result.
  - A projection head trained with listing identity as the positive raised catalogue **test R@1 from 0.316 to 0.434**, but lowered own-photo holdout **graded nDCG@5 from 0.90 to 0.64** (44 of 50 items worse).
  - An augmentation-only variant was also worse on tune.
  - Both doubled the share of wrong-category look-alikes in the top-5.
  - Same-listing Recall@k rewards learning each shop's photography style, which does not help find *similar designs* for a piece that is not in the catalogue. Without the own-photo graded set this would have been shipped as a 37% improvement.
- **MMR for photo search (Phase 8).** It helped catalogue-to-catalogue retrieval but cut own-photo nDCG@5 by 0.093 (tune −0.039), because the best look-alikes for a real photo are often near-duplicates of each other. It is off for photo search and on for "more like this".

### C5. Remaining failure cases (with evidence)

- **Undercounting multi-item photos.** Count exact is 0.63; *overlap*-tagged photos 0.09 (n=11) and *chain* 0.00 (n=4). Small items (earrings) inside a necklace's loop get no box; tiling recovered some but added as many false boxes.
- **Earrings rejected as out of scope.** All 4 holdout false rejects (6.3%) are earrings with high similarity (0.57–0.77). Hoops and studs read as rings to CLIP.
- **Out-of-scope jewellery still accepted.** 3 of the 4 holdout rings, brooches and anklets get through. There are only 4 such holdout photos (CI 0.25–1.00); more are needed to calibrate properly.
- **Multi-piece "set" listings.** 352 listings are sets with a single category label. They lower category precision and occasionally surface for single-item queries.
- **Prices** use fixed exchange rates (USD 88, GBP 117) set on 2026-10-12.

### C6. Caveats on the evidence

- **Holdout is mildly optimistic.** The failure modes that motivated v2 (chains, overlapping earrings, pendants) had been seen across the whole stumper set before it was split.
- **Holdout was looked at more than once.** It was used once each to evaluate the rejected FT head and MMR on photos; neither was adopted on holdout evidence, and both decisions agree with tune.
- **Small samples.** 73 holdout photos (9 negatives), and one grader. CIs are wide, especially for negatives.
- **Grade drift.** The same grader applied "3 = near-identical" more strictly in a later blind session (23% vs 83% of judgements). The "Closest matches" tier is therefore defined on grade ≥ 2, which was consistent, and when a result was graded twice the first grade is kept.
- **Exact vs approximate search.** Exact search was used for embedding comparisons; it matched production HNSW exactly for DINOv2 (val R@1 0.297 both).

### C7. Running and reproducing

See `README.md` (section "v2") for run commands, the evaluation scripts and where each result lives. The full numeric log is in `results/LOG.md`.

> 🙋 **Screenshots for this writeup:** about 5 per category of the final UI (upload plus results) are still to be added. Run `PORT=5001 python src/api/app.py` and open http://localhost:5001.
