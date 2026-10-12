"""
Phase 1 — retrieval evaluation harness (same-listing protocol).

For every eligible listing in a split, its fixed query image (data/splits.json)
is the query; the relevant item is the same catalogue_id. The query image
itself is in the index, so it is excluded from results by faiss id. This is
non-circular: relevance comes from listing identity, never from a model.

Modes
  retrieval  : embed the whole image -> search. All eligible queries (clean,
               stored embedding) + a fixed 300-query subset per augmentation
               (re-embedded, so clean vs augmented is like-for-like).
  pipeline   : preprocess -> OWL-ViT detect -> crop -> embed -> classify ->
               search -> rerank -> decide, on the same 300-query subset,
               clean and "combined" augmentation. Scored on the top-scoring
               detection (as the user would see first) and best-of-items.

Ranking variants (all modes report them)
  raw        : FAISS search + per-listing dedup, no re-rank.
  prod       : + the live classifier re-rank, fed the L2-normalised embedding
               exactly as src/pipeline does today  <- the deployed baseline.
  fixed_input: DIAGNOSTIC — same classifier fed the un-normalised embedding it
               was trained on. Sizes the Phase 0 finding; not a tuned change.

Metrics: Recall@1/5/10, MRR (over the top 20), nDCG@5 (binary), graded
nDCG@5 (3 same listing / 2 same category + same non-blank subtype / 1 same
category — PROVISIONAL until Phase 2 backfills subtype), category P@5
(non-set queries), and a pHash variant where listings holding a near-duplicate
(Hamming <= 6) of the query image also count as relevant.

Usage
  python src/eval/retrieval_eval.py --split val
  python src/eval/retrieval_eval.py --split test        # once only; refuses a rerun
  flags: --tag baseline  --n-sub 300  --skip-aug  --skip-pipeline
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
import torch

sys.path.append(".")
import config
from src.eval.common import (SPLITS_PATH, PHASH_CACHE_PATH, RESULTS_DIR, Progress, bootstrap_ci,
                             fmt_ci, load_json, save_json, thumb_b64, open_image, replace_log_section)
from src.eval.augment import ALL_AUGS, AUG_MAX_SIDE, apply_aug
from src.pipeline.embed import load_embedder
from src.pipeline.classify import load_classifier
from src.pipeline.retrieve import apply_rerank
from src.pipeline.detect import detect, crop_with_padding, load_detector
from src.pipeline.preprocess import preprocess_image
from src.pipeline.decision import decide

TOP_K = 20            # same as the API
RAW_K = 40            # same as src/pipeline/retrieve.search
PHASH_NEAR_DUP = 6
VARIANTS = ["raw", "prod", "fixed_input"]
SEED = 42
EMBED_BATCH = 16


# ----------------------------------------------------------------- context

class Ctx:
    def __init__(self, split, metadata_path=config.CATALOGUE_METADATA_PATH):
        self.split = split
        self.metadata_path = metadata_path
        splits = load_json(SPLITS_PATH)["listings"]
        self.listings = splits
        self.emb_meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
        self.fid_of_path = {p: i for i, p in enumerate(self.emb_meta["image_path"])}
        self.cid_of_fid = self.emb_meta["catalogue_id"].to_numpy()
        self.cat_of_fid = self.emb_meta["category"].to_numpy()
        self.raw_emb = np.load(config.CATALOGUE_EMBEDDINGS_PATH, mmap_mode="r")

        meta = pd.read_parquet(metadata_path)
        self.color = dict(zip(meta["catalogue_id"], meta["color_query"].fillna("").astype(str).str.strip()))
        self.subtype = dict(zip(meta["catalogue_id"], meta["subtype"].fillna("").astype(str).str.strip()))
        # "trusted" = labels not predicted by an image model (original / variant / title). Model-filled
        # labels come from CLIP+DINOv2 features and would flatter DINOv2-like retrievers.
        model_src = {"clip", "probe"}
        self.subtype_trusted = ({c: (v if s not in model_src else "") for c, v, s in
                                 zip(meta["catalogue_id"], meta["subtype"].fillna("").str.strip(), meta["subtype_source"])}
                                if "subtype_source" in meta else dict(self.subtype))
        self.color_trusted = ({c: (v if s not in model_src else "") for c, v, s in
                               zip(meta["catalogue_id"], meta["color_query"].fillna("").str.strip(), meta["color_source"])}
                              if "color_source" in meta else dict(self.color))
        self.title = dict(zip(meta["catalogue_id"], meta["title"]))
        self.category = dict(zip(meta["catalogue_id"], meta["category"]))
        self.n_cat = meta["category"].value_counts().to_dict()
        self.n_cat_sub = self._cat_sub_counts(self.subtype)
        self.n_cat_sub_trusted = self._cat_sub_counts(self.subtype_trusted)

        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        self.clf, self.classes = load_classifier()

        ph = pd.read_parquet(PHASH_CACHE_PATH) if os.path.exists(PHASH_CACHE_PATH) else None
        self.phash = ph["phash"].to_numpy(dtype=np.uint64) if ph is not None else None

        self.queries = [dict(cid=cid, path=v["query_image"], fid=self.fid_of_path[v["query_image"]],
                             category=v["category"], is_set=v["is_set"], subtype=self.subtype.get(cid, ""),
                             color=self.color.get(cid, ""))
                        for cid, v in sorted(splits.items()) if v["split"] == split and v["eligible"]]

    def _cat_sub_counts(self, subtype):
        out = defaultdict(int)
        for cid, s in subtype.items():
            if s:
                out[(self.category[cid], s)] += 1
        return dict(out)

    def subset(self, n):
        """Fixed category-stratified subset (same for every run of this split)."""
        rng = np.random.default_rng(SEED)
        by_cat = defaultdict(list)
        for q in self.queries:
            by_cat[q["category"]].append(q)
        if n >= len(self.queries):
            return list(self.queries)
        out = []
        for cat, qs in sorted(by_cat.items()):
            k = round(n * len(qs) / len(self.queries))
            out += [qs[i] for i in sorted(rng.choice(len(qs), size=min(k, len(qs)), replace=False))]
        return out

    def near_dup_listings(self, fid):
        if self.phash is None:
            return set()
        ham = np.bitwise_count(self.phash ^ self.phash[fid])
        hits = np.where(ham <= PHASH_NEAR_DUP)[0]
        return {self.cid_of_fid[h] for h in hits if h != fid}


# ----------------------------------------------------------------- search + rank

def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def search_excluding(ctx, q_norm, exclude_fid):
    """Mirror of src/pipeline/retrieve.search, with the query image removed."""
    t0 = time.perf_counter()
    scores, ids = ctx.index.search(q_norm.reshape(1, -1).astype(np.float32), RAW_K + 1)
    t_search = time.perf_counter() - t0
    best = {}
    for s, fid in zip(scores[0], ids[0]):
        if fid == -1 or fid == exclude_fid:
            continue
        cid = ctx.cid_of_fid[fid]
        if cid not in best or s > best[cid]["score"]:
            best[cid] = {"catalogue_id": cid, "category": ctx.cat_of_fid[fid],
                         "image_path": ctx.emb_meta["image_path"].iat[fid], "score": float(s)}
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:TOP_K]
    return ranked, t_search


def class_probs(ctx, x):
    p = ctx.clf.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
    return {c: float(v) for c, v in zip(ctx.classes, p)}


def rank_variants(ctx, raw_vec, exclude_fid):
    """raw_vec: un-normalised DINOv2 CLS. Returns {variant: ranked list}, timings."""
    qn = l2n(raw_vec)
    ranked, t_search = search_excluding(ctx, qn, exclude_fid)
    t0 = time.perf_counter()
    probs_prod = class_probs(ctx, qn)
    prod = apply_rerank(ranked, probs_prod)
    t_rerank = time.perf_counter() - t0
    fixed = apply_rerank(ranked, class_probs(ctx, raw_vec))
    return ({"raw": ranked, "prod": prod, "fixed_input": fixed},
            {"search": t_search, "classify_rerank": t_rerank},
            {"prod": max(probs_prod, key=probs_prod.get),
             "fixed_input": max(class_probs(ctx, raw_vec).items(), key=lambda kv: kv[1])[0]})


# ----------------------------------------------------------------- metrics

def graded_gain(ctx, q, r, subtype=None):
    subtype = subtype if subtype is not None else ctx.subtype
    if r["catalogue_id"] == q["cid"]:
        return 3
    if r["category"] != q["category"]:
        return 0
    qs = subtype.get(q["cid"], "")
    if qs and subtype.get(r["catalogue_id"], "") == qs:
        return 2
    return 1


def ideal_graded_dcg(ctx, q, subtype=None, n_cat_sub=None):
    subtype = subtype if subtype is not None else ctx.subtype
    n_cat_sub = n_cat_sub if n_cat_sub is not None else ctx.n_cat_sub
    qs = subtype.get(q["cid"], "")
    n2 = n_cat_sub.get((q["category"], qs), 1) - 1 if qs else 0
    n1 = ctx.n_cat[q["category"]] - 1 - n2
    gains = ([3] + [2] * n2 + [1] * n1)[:5]
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def score(ctx, q, ranked, near_dups=None):
    ids = [r["catalogue_id"] for r in ranked]
    rank = ids.index(q["cid"]) + 1 if q["cid"] in ids else None
    m = {
        "rank": rank,
        "r1": float(rank is not None and rank <= 1),
        "r5": float(rank is not None and rank <= 5),
        "r10": float(rank is not None and rank <= 10),
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg5": 1.0 / np.log2(rank + 1) if rank and rank <= 5 else 0.0,
        "graded_ndcg5": sum(graded_gain(ctx, q, r) / np.log2(i + 2) for i, r in enumerate(ranked[:5]))
                        / ideal_graded_dcg(ctx, q),
        "cat_p5": float(np.mean([r["category"] == q["category"] for r in ranked[:5]])) if ranked else 0.0,
        "top1_score": ranked[0]["score"] if ranked else None,
    }
    m["graded_ndcg5_trusted"] = (sum(graded_gain(ctx, q, r, ctx.subtype_trusted) / np.log2(i + 2) for i, r in enumerate(ranked[:5]))
                                 / ideal_graded_dcg(ctx, q, ctx.subtype_trusted, ctx.n_cat_sub_trusted))
    # attribute precision@5: only for queries whose own attribute is known; blank results count as non-matching
    for name, col, sub in (("color_p5", ctx.color, False), ("subtype_p5", ctx.subtype, True),
                           ("color_p5_trusted", ctx.color_trusted, False), ("subtype_p5_trusted", ctx.subtype_trusted, True)):
        qv = col.get(q["cid"], "")
        if qv and ranked:
            m[name] = float(np.mean([col.get(r["catalogue_id"], "") == qv and (not sub or r["category"] == q["category"])
                                     for r in ranked[:5]]))
    if near_dups is not None:
        rel = near_dups | {q["cid"]}
        rank_ph = next((i + 1 for i, c in enumerate(ids) if c in rel), None)
        m.update({"r1_phash": float(rank_ph is not None and rank_ph <= 1),
                  "r5_phash": float(rank_ph is not None and rank_ph <= 5),
                  "mrr_phash": 1.0 / rank_ph if rank_ph else 0.0,
                  "has_cross_listing_dup": float(len(near_dups - {q["cid"]}) > 0)})
    return m


METRIC_KEYS = ["r1", "r5", "r10", "mrr", "ndcg5", "graded_ndcg5", "graded_ndcg5_trusted", "cat_p5",
               "color_p5", "subtype_p5", "color_p5_trusted", "subtype_p5_trusted"]
PHASH_KEYS = ["r1_phash", "r5_phash", "mrr_phash", "has_cross_listing_dup"]
CI_KEYS = ["r1", "r5", "mrr", "ndcg5"]


def aggregate(rows, with_ci=False):
    """rows: list of (query, metrics). Returns means (+CIs) for all / non_set / per category."""
    def agg(sel):
        if not sel:
            return {"n": 0}
        out = {"n": len(sel)}
        for k in METRIC_KEYS + PHASH_KEYS:
            vals = [m[k] for _, m in sel if k in m]
            if vals:
                out[k] = float(np.mean(vals))
                if k.startswith(("color_p5", "subtype_p5")):
                    out[f"n_{k}"] = len(vals)
        if with_ci:
            out["ci"] = {k: bootstrap_ci([m[k] for _, m in sel]) for k in CI_KEYS}
        return out

    non_set = [(q, m) for q, m in rows if not q["is_set"]]
    res = {"all": agg(rows), "non_set": agg(non_set)}
    # category P@5 headline is computed on non-set queries only (set listings are multi-category)
    res["all"]["cat_p5_non_set"] = res["non_set"].get("cat_p5")
    res["by_category"] = {c: agg([(q, m) for q, m in non_set if q["category"] == c])
                          for c in config.CATEGORIES}
    res["set_only"] = agg([(q, m) for q, m in rows if q["is_set"]])
    return res


def pct(values):
    v = np.asarray(values, dtype=float) * 1000
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "n": int(len(v))} if len(v) else {}


# ----------------------------------------------------------------- embedding

def embed_batch_raw(images):
    processor, model = load_embedder()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].numpy()


# ----------------------------------------------------------------- modes

def run_retrieval_clean(ctx):
    print(f"\n[retrieval/clean] {len(ctx.queries)} queries (stored embeddings)")
    rows = {v: [] for v in VARIANTS}
    per_query = []
    t_search, t_rerank = [], []
    cls_ok = {"prod": [], "fixed_input": []}
    prog = Progress(len(ctx.queries), "clean")
    for i, q in enumerate(ctx.queries):
        raw = np.asarray(ctx.raw_emb[q["fid"]], dtype=np.float32)
        ranked, times, pred = rank_variants(ctx, raw, q["fid"])
        nd = ctx.near_dup_listings(q["fid"])
        for v in VARIANTS:
            rows[v].append((q, score(ctx, q, ranked[v], nd)))
        for k in cls_ok:
            if not q["is_set"]:
                cls_ok[k].append(float(pred[k] == q["category"]))
        t_search.append(times["search"])
        t_rerank.append(times["classify_rerank"])
        per_query.append({"q": q, "prod": ranked["prod"][:5], "metrics": rows["prod"][-1][1]})
        prog.update(i + 1)
    result = {v: aggregate(rows[v], with_ci=True) for v in VARIANTS}
    result["latency"] = {"search": pct(t_search), "classify_rerank": pct(t_rerank)}
    result["classifier_on_query_image_acc_non_set"] = {k: float(np.mean(v)) for k, v in cls_ok.items()}
    return result, per_query


def run_retrieval_aug(ctx, subset, augs):
    out = {}
    t_embed = []
    for aug in augs:
        rows = {v: [] for v in VARIANTS}
        prog = Progress(len(subset), f"aug:{aug}")
        for b in range(0, len(subset), EMBED_BATCH):
            batch = subset[b:b + EMBED_BATCH]
            imgs = [apply_aug(aug, open_image(q["path"], AUG_MAX_SIDE), q["cid"]) for q in batch]
            t0 = time.perf_counter()
            raws = embed_batch_raw(imgs)
            t_embed.append((time.perf_counter() - t0) / len(batch))
            for q, raw in zip(batch, raws):
                ranked, _, _ = rank_variants(ctx, raw, q["fid"])
                nd = ctx.near_dup_listings(q["fid"])
                for v in VARIANTS:
                    rows[v].append((q, score(ctx, q, ranked[v], nd)))
            prog.update(min(b + EMBED_BATCH, len(subset)))
        out[aug] = {v: aggregate(rows[v]) for v in VARIANTS}
        a = out[aug]["prod"]["all"]
        print(f"    {aug:16s} prod R@1={a['r1']:.3f} R@5={a['r5']:.3f} MRR={a['mrr']:.3f}")
    return out, {"embed_amortised_batch16": pct(t_embed)}


def run_pipeline_mode(ctx, subset, augs=("clean", "combined")):
    load_detector()
    out, lat, examples = {}, defaultdict(list), {}
    for aug in augs:
        rows_top = {v: [] for v in VARIANTS}
        rows_best = {v: [] for v in VARIANTS}
        n_items, n_detected, det_cat_ok = [], 0, []
        prog = Progress(len(subset), f"pipeline:{aug}")
        for i, q in enumerate(subset):
            img = apply_aug(aug, open_image(q["path"]), q["cid"]) if aug != "clean" else open_image(q["path"])
            t0 = time.perf_counter(); img = preprocess_image(img); lat["preprocess"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); dets = detect(img); lat["detect"].append(time.perf_counter() - t0)
            n_items.append(len(dets))
            nd = ctx.near_dup_listings(q["fid"])
            item_scores = []
            for d in dets:
                t0 = time.perf_counter(); crop = crop_with_padding(img, d["box"]); lat["crop"].append(time.perf_counter() - t0)
                # same forward pass as src/pipeline/embed.embed_image, kept un-normalised so the
                # fixed_input diagnostic can use it; rank_variants normalises for search/prod.
                t0 = time.perf_counter(); raw = embed_batch_raw([crop])[0]; lat["embed"].append(time.perf_counter() - t0)
                ranked, times, _ = rank_variants(ctx, raw, q["fid"])
                lat["search"].append(times["search"]); lat["classify_rerank"].append(times["classify_rerank"])
                t0 = time.perf_counter(); decide(ranked["prod"]); lat["decide"].append(time.perf_counter() - t0)
                item_scores.append((d, {v: score(ctx, q, ranked[v], nd) for v in VARIANTS}))
            if dets:
                n_detected += 1
                top = max(item_scores, key=lambda t: t[0]["score"])
                det_cat_ok.append(float(top[0]["category"] == q["category"]))
            for v in VARIANTS:
                if dets:
                    rows_top[v].append((q, top[1][v]))
                    rows_best[v].append((q, max((s[v] for _, s in item_scores), key=lambda m: (m["mrr"], m["cat_p5"]))))
                else:
                    miss = {k: 0.0 for k in METRIC_KEYS + ["r1_phash", "r5_phash", "mrr_phash"]
                            if not k.startswith(("color_p5", "subtype_p5"))}
                    miss.update(rank=None, top1_score=None, has_cross_listing_dup=0.0)
                    rows_top[v].append((q, miss)); rows_best[v].append((q, miss))
            prog.update(i + 1)
        out[aug] = {"top_detection": {v: aggregate(rows_top[v], with_ci=True) for v in VARIANTS},
                    "best_of_items": {v: aggregate(rows_best[v]) for v in VARIANTS},
                    "detection_rate": n_detected / len(subset),
                    "mean_items_per_image": float(np.mean(n_items)),
                    "detector_label_acc_top_item": float(np.mean(det_cat_ok)) if det_cat_ok else None}
        a = out[aug]["top_detection"]["prod"]["all"]
        print(f"    pipeline {aug:10s} prod R@1={a['r1']:.3f} R@5={a['r5']:.3f} det_rate={out[aug]['detection_rate']:.3f}")
    return out, {k: pct(v) for k, v in lat.items()}


# ----------------------------------------------------------------- report

def worst_queries(ctx, per_query, n=20):
    def key(pq):
        r = pq["metrics"]["rank"]
        return (0 if r is None else 1, -(r or 0), -(pq["prod"][0]["score"] if pq["prod"] else 0))
    worst = sorted(per_query, key=key)[:n]
    out = []
    for pq in worst:
        q = pq["q"]
        others = [p for p in ctx.emb_meta.loc[ctx.emb_meta.catalogue_id == q["cid"], "image_path"] if p != q["path"]]
        out.append({"query_image": q["path"], "catalogue_id": q["cid"], "category": q["category"],
                    "is_set": q["is_set"], "title": ctx.title.get(q["cid"], ""), "rank": pq["metrics"]["rank"],
                    "same_listing_other_image": others[0] if others else None,
                    "top3": [{"catalogue_id": r["catalogue_id"], "category": r["category"], "score": r["score"],
                              "image_path": r["image_path"], "title": ctx.title.get(r["catalogue_id"], "")}
                             for r in pq["prod"][:3]]})
    return out


def table_row(name, a):
    return (f"| {name} | {a['n']} | {a.get('r1', 0):.3f} | {a.get('r5', 0):.3f} | {a.get('r10', 0):.3f} | "
            f"{a.get('mrr', 0):.3f} | {a.get('ndcg5', 0):.3f} | {a.get('graded_ndcg5', 0):.3f} | "
            f"{(a.get('cat_p5_non_set') or a.get('cat_p5') or 0):.3f} | {a.get('r1_phash', 0):.3f} | {a.get('r5_phash', 0):.3f} | "
            f"{a.get('color_p5', float('nan')):.3f} | {a.get('subtype_p5', float('nan')):.3f} | "
            f"{a.get('graded_ndcg5_trusted', float('nan')):.3f} | {a.get('color_p5_trusted', float('nan')):.3f} | "
            f"{a.get('subtype_p5_trusted', float('nan')):.3f} |")


HDR = ("| variant | n | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 | cat P@5 (non-set) | R@1 pHash | R@5 pHash | colour P@5 | subtype P@5 | graded nDCG@5 (trusted) | colour P@5 (trusted) | subtype P@5 (trusted) |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def markdown_section(res, tag, split):
    c = res["retrieval_clean"]
    lines = [f"## {res.get('phase', 'Phase 1')} — {tag}, split = {split} ({res['created']})", "",
             "Same-listing protocol: query image excluded from the index. Variant `prod` is what is deployed now. "
             "`fixed_input` is a diagnostic (classifier fed the un-normalised embedding it was trained on), not a tuned change. "
             f"Metadata: `{res.get('metadata', config.CATALOGUE_METADATA_PATH)}` — graded nDCG@5: {res['notes']['graded_ndcg5']}. "
             "Colour/subtype P@5 are over queries whose own attribute is known; blank results count as non-matching. "
             "*(trusted)* columns use only original/variant/title labels (no image-model fills) — use these to compare retrievers.", "",
             "**Retrieval-only, clean, all eligible queries**", "", HDR]
    for v in VARIANTS:
        lines.append(table_row(f"{v} (all)", c[v]["all"]))
    for v in VARIANTS:
        lines.append(table_row(f"{v} (non-set)", c[v]["non_set"]))
    ci = c["prod"]["all"]["ci"]
    lines += ["", f"prod 95% bootstrap CI: R@1 {fmt_ci(ci['r1'])}, R@5 {fmt_ci(ci['r5'])}, MRR {fmt_ci(ci['mrr'])}",
              f"Queries whose image has a cross-listing pHash near-duplicate: {c['prod']['all'].get('has_cross_listing_dup', 0):.3f}",
              f"Classifier accuracy on the query image (non-set): prod input {c['classifier_on_query_image_acc_non_set']['prod']:.3f}, "
              f"fixed input {c['classifier_on_query_image_acc_non_set']['fixed_input']:.3f}", "",
              "**Per category (prod, non-set)**", "", HDR]
    for cat in config.CATEGORIES:
        lines.append(table_row(cat, c["prod"]["by_category"][cat]))
    if "retrieval_aug" in res:
        lines += ["", f"**Per augmentation (retrieval-only, re-embedded, fixed {res['n_subset']}-query subset)**", "",
                  "| augmentation | R@1 raw | R@1 prod | R@1 fixed | R@5 prod | MRR prod | cat P@5 prod |", "|---|---|---|---|---|---|---|"]
        for aug, d in res["retrieval_aug"].items():
            lines.append(f"| {aug} | {d['raw']['all']['r1']:.3f} | {d['prod']['all']['r1']:.3f} | {d['fixed_input']['all']['r1']:.3f} | "
                         f"{d['prod']['all']['r5']:.3f} | {d['prod']['all']['mrr']:.3f} | {d['prod']['all']['cat_p5_non_set']:.3f} |")
    if "pipeline" in res:
        lines += ["", f"**Full pipeline (detect → crop → embed → retrieve), {res['n_subset']}-query subset, scored on top detection**", "",
                  "| input | det. rate | items/img | detector label acc | R@1 prod | R@5 prod | MRR prod | R@1 fixed | R@5 best-of-items (prod) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for aug, d in res["pipeline"].items():
            t = d["top_detection"]
            lines.append(f"| {aug} | {d['detection_rate']:.3f} | {d['mean_items_per_image']:.2f} | {d['detector_label_acc_top_item'] or 0:.3f} | "
                         f"{t['prod']['all']['r1']:.3f} | {t['prod']['all']['r5']:.3f} | {t['prod']['all']['mrr']:.3f} | "
                         f"{t['fixed_input']['all']['r1']:.3f} | {d['best_of_items']['prod']['all']['r5']:.3f} |")
    lines += ["", "**Latency (ms)**", "", "| stage | p50 | p95 | n |", "|---|---|---|---|"]
    for k, v in res["latency"].items():
        if v:
            lines.append(f"| {k} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['n']} |")
    return "\n".join(lines)


def write_html(res, worst, path, tag, split):
    md = markdown_section(res, tag, split)
    # minimal markdown -> html for tables/paragraphs (self-contained, no deps)
    html_lines, in_table = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            tag_ = "th" if not in_table else "td"
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            html_lines.append("<tr>" + "".join(f"<{tag_}>{c}</{tag_}>" for c in cells) + "</tr>")
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if line.startswith("## "):
            html_lines.append(f"<h1>{line[3:]}</h1>")
        elif line.startswith("**") and line.endswith("**"):
            html_lines.append(f"<h3>{line.strip('*')}</h3>")
        elif line:
            html_lines.append(f"<p>{line.replace('**', '')}</p>")
    if in_table:
        html_lines.append("</table>")

    cards = []
    for w in worst:
        res_cells = "".join(
            f'<div class="t"><img src="{thumb_b64(r["image_path"])}"><div>{r["category"]} · {r["score"]:.3f}<br>'
            f'<small>{r["title"][:50]}</small></div></div>' for r in w["top3"])
        other = (f'<div class="t gt"><img src="{thumb_b64(w["same_listing_other_image"])}"><div>same listing (target)</div></div>'
                 if w["same_listing_other_image"] else "")
        cards.append(f'<div class="row"><div class="t q"><img src="{thumb_b64(w["query_image"])}"><div><b>query</b> · {w["category"]}'
                     f'{" · SET" if w["is_set"] else ""}<br>rank: {w["rank"] or "&gt;20"}<br><small>{w["title"][:50]}</small><br>'
                     f'<code>{w["query_image"]}</code></div></div>{other}<div class="arrow">→</div>{res_cells}</div>')

    css = """body{font-family:-apple-system,sans-serif;margin:24px;color:#222;max-width:1200px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}
th{background:#f6f0fa}td:first-child,th:first-child{text-align:left}.row{display:flex;gap:10px;align-items:flex-start;
border-bottom:1px solid #eee;padding:10px 0}.t{width:140px;font-size:11px}.t img{width:130px;height:130px;object-fit:cover;
border:1px solid #ccc;border-radius:6px}.q img{border:2px solid #7b5ea7}.gt img{border:2px solid #2e7d32}.arrow{font-size:28px;
align-self:center;color:#999}code{font-size:9px;word-break:break-all}"""
    with open(path, "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Jewel Lens {tag} ({split})</title><style>{css}</style></head><body>"
                + "\n".join(html_lines) + "<h2>Worst 20 queries (retrieval-only, clean, prod)</h2>" + "".join(cards) + "</body></html>")


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n-sub", type=int, default=300)
    ap.add_argument("--skip-aug", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--phase", default="Phase 1", help="label used in the LOG.md header")
    ap.add_argument("--metadata", default=config.CATALOGUE_METADATA_PATH,
                    help="metadata parquet for subtype/colour (graded nDCG, attribute P@5)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{args.tag}.json")
    all_res = load_json(json_path) if os.path.exists(json_path) else {}
    if args.split == "test" and "test" in all_res:
        sys.exit(f"Test split already evaluated for '{args.tag}' ({all_res['test']['created']}). "
                 "Test is run once per frozen variant; use a new --tag for a new variant.")

    t_start = time.time()
    ctx = Ctx(args.split, args.metadata)
    if ctx.phash is None:
        print(f"[warn] {PHASH_CACHE_PATH} missing — run src/eval/phash_cache.py first; pHash variant skipped.")
    subset = ctx.subset(args.n_sub)
    print(f"Split={args.split}: {len(ctx.queries)} eligible queries "
          f"({sum(q['is_set'] for q in ctx.queries)} sets), subset={len(subset)}")

    res = {"created": datetime.now().isoformat(timespec="seconds"), "split": args.split,
           "n_queries": len(ctx.queries), "n_subset": len(subset), "metadata": args.metadata, "phase": args.phase,
           "notes": {"graded_ndcg5": ("PROVISIONAL — subtype is ~60% blank until Phase 2" if args.metadata == config.CATALOGUE_METADATA_PATH
                                          else f"subtype from {args.metadata}"),
                     "fixed_input": "diagnostic only (classifier fed un-normalised embedding)",
                     "pipeline_preprocess": "eval pipeline applies preprocess_image(); the live API currently does not",
                     "phash_variant": f"listings holding an image within Hamming <= {PHASH_NEAR_DUP} of the query image also count as relevant"}}

    clean, per_query = run_retrieval_clean(ctx)
    res["retrieval_clean"] = clean
    latency = dict(clean.pop("latency"))
    a = clean["prod"]["all"]
    print(f"  clean prod: R@1={a['r1']:.3f} R@5={a['r5']:.3f} MRR={a['mrr']:.3f} nDCG@5={a['ndcg5']:.3f} catP@5(non-set)={a['cat_p5_non_set']:.3f}")

    if not args.skip_aug:
        print(f"\n[retrieval/augmented] {len(ALL_AUGS)} conditions × {len(subset)} queries")
        res["retrieval_aug"], lat_aug = run_retrieval_aug(ctx, subset, ALL_AUGS)
        latency.update(lat_aug)

    if not args.skip_pipeline:
        print(f"\n[pipeline] clean + combined × {len(subset)} queries")
        res["pipeline"], lat_pipe = run_pipeline_mode(ctx, subset)
        latency.update({f"pipeline_{k}": v for k, v in lat_pipe.items()})

    res["latency"] = latency
    worst = worst_queries(ctx, per_query)
    res["worst20"] = worst
    res["runtime_min"] = (time.time() - t_start) / 60

    all_res[args.split] = res
    save_json(all_res, json_path)
    report_path = os.path.join(RESULTS_DIR, f"{args.tag}_report{'' if args.split == 'val' else '_' + args.split}.html")
    write_html(res, worst, report_path, args.tag, args.split)
    if not args.no_log:
        replace_log_section(f"## {args.phase} — {args.tag}, split = {args.split} ", markdown_section(res, args.tag, args.split))
    print(f"\nWrote {json_path}, {report_path}{'' if args.no_log else ', appended results/LOG.md'} "
          f"in {res['runtime_min']:.1f} min")


if __name__ == "__main__":
    main()
