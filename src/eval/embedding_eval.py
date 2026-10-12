"""
Phase 6 — compare embedding variants on the Phase 1 same-listing queries (val).

Variants (image-level cosine similarity, then per-listing aggregation, query image excluded)
  A  dino        DINOv2-base CLS (current production embedding)
  B  dino_nobg   DINOv2 of background-removed images (rembg/u2net onto white), index AND query
  C  clip        CLIP ViT-B/32 image embedding
  D  dino+clip   w * cos_dino + (1 - w) * cos_clip  (= concatenating sqrt(w)·dino, sqrt(1-w)·clip);
                 w chosen on val
  (B+C)          if B is available: w * cos_nobg + (1 - w) * cos_clip
  E  best + listing aggregation: score of a listing = max (v1) / mean / 0.5·max + 0.5·mean of
                 its images' similarities
Each is reported without re-rank and with the Phase 4 fused re-rank (production since Phase 4).
Exact search (matrix product) is used for all variants so they are comparable; the HNSW
approximation gap for A is reported against Phase 1's numbers.

Primary metric (for choosing on val): mean of MRR(clean, all queries) and MRR(combined
phone-photo augmentation, 300-query subset). Secondary: R@k, nDCG@5, category P@5,
trusted graded nDCG@5 (Phase 2 labels from title/variant only), per augmentation.

Usage: python src/eval/embedding_eval.py [--split val]  -> results/phase6.json, LOG.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import time
from datetime import datetime

import numpy as np

sys.path.append(".")
import config
from src.eval.common import save_json, fmt_ci, replace_log_section, load_json
from src.eval.augment import ALL_AUGS
from src.eval.retrieval_eval import Ctx, score, aggregate
from src.pipeline.retrieve import apply_rerank
from src.pipeline.category import categorize, load_category_models

CACHE = "artifacts/eval_cache"
BACKFILLED = "data/catalogue_metadata_backfilled.parquet"
TOP_K = 20
W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
AGGS = ["max", "mean", "maxmean"]


def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


class Index:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mats = {"dino": l2n(np.load(config.CATALOGUE_EMBEDDINGS_PATH)),
                     "clip": l2n(np.load("artifacts/clip_b32/embeddings.npy"))}
        if os.path.exists("artifacts/dinov2_nobg/embeddings.npy"):
            self.mats["nobg"] = l2n(np.load("artifacts/dinov2_nobg/embeddings.npy"))
        cids = ctx.cid_of_fid
        # images of a listing are contiguous in the meta order -> reduceat segments
        starts = np.r_[0, np.where(cids[1:] != cids[:-1])[0] + 1]
        assert len(starts) == len(set(cids)), "listing images are not contiguous"
        self.starts, self.counts = starts, np.diff(np.r_[starts, len(cids)])
        self.listing_of_fid = np.repeat(np.arange(len(starts)), self.counts)
        self.cid = cids[starts]
        self.cat = ctx.cat_of_fid[starts]

    def rank(self, img_scores, exclude_fid, agg):
        """exclude_fid: the query's own image (same-listing protocol) or None (an outside photo)."""
        s = img_scores.astype(np.float64).copy()
        s_max = s.copy()
        if exclude_fid is not None:
            s_max[exclude_fid] = -np.inf
        mx = np.maximum.reduceat(s_max, self.starts)
        if agg == "max":
            ls = mx
        else:
            s_sum = s.copy()
            cnt = self.counts.astype(np.float64).copy()
            if exclude_fid is not None:
                s_sum[exclude_fid] = 0.0
                cnt[self.listing_of_fid[exclude_fid]] -= 1
            mean = np.add.reduceat(s_sum, self.starts) / np.maximum(cnt, 1)
            mean[cnt == 0] = -np.inf
            ls = mean if agg == "mean" else 0.5 * mx + 0.5 * mean
        top = np.argpartition(-ls, TOP_K)[:TOP_K]
        top = top[np.argsort(-ls[top])]
        return [{"catalogue_id": self.cid[i], "category": self.cat[i], "score": float(ls[i])} for i in top]


def query_sets(ctx, index):
    """{aug: list of (query, {feature: vector})}; clean uses the stored index rows."""
    out = {"clean": [(q, {k: m[q["fid"]] for k, m in index.mats.items()}) for q in ctx.queries]}
    raw_dino = np.load(config.CATALOGUE_EMBEDDINGS_PATH, mmap_mode="r")
    out["_clean_raw"] = {q["cid"]: np.asarray(raw_dino[q["fid"]], np.float32) for q in ctx.queries}
    for aug in [a for a in ALL_AUGS if a != "clean"]:
        path = os.path.join(CACHE, f"{ctx.split}_{aug}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        feats = {"dino": l2n(z["dino"]), "clip": l2n(z["clip"]), "dino_raw": z["dino"]}
        if "nobg" in z:
            feats["nobg"] = l2n(z["nobg"])
        out[aug] = [(ctx.queries[qi], {k: v[i] for k, v in feats.items()}) for i, qi in enumerate(z["query_index"])]
    return out


def variant_scores(index, feats, spec):
    """spec: {'dino': w1, 'clip': w2, 'nobg': w3} -> image-level score vector."""
    return sum(w * (index.mats[k] @ feats[k]) for k, w in spec.items() if w)


def fused_probs(index, ctx, q, feats, raw):
    """Phase 4 fused category of the query image (kNN over exact DINOv2 neighbours + CLIP + clf_v2)."""
    nb = index.rank(index.mats["dino"] @ feats["dino"], q["fid"], "max")
    return categorize(None, feats["dino"], raw, "earring", nb, clip_emb=feats["clip"])["probs"]


def evaluate(index, ctx, qsets, spec, agg, probs_cache):
    res = {}
    for aug, items in qsets.items():
        if aug.startswith("_"):
            continue
        rows = {"none": [], "fused": []}
        for q, f in items:
            if any(k not in f for k, w in spec.items() if w):
                break
            ranked = index.rank(variant_scores(index, f, spec), q["fid"], agg)
            key = (aug, q["cid"])
            if key not in probs_cache:
                raw = qsets["_clean_raw"][q["cid"]] if aug == "clean" else f["dino_raw"]
                probs_cache[key] = fused_probs(index, ctx, q, f, raw)
            rows["none"].append((q, score(ctx, q, ranked)))
            rows["fused"].append((q, score(ctx, q, apply_rerank([dict(r) for r in ranked], probs_cache[key]))))
        else:
            res[aug] = {k: aggregate(v, with_ci=(aug == "clean")) for k, v in rows.items()}
    return res


def primary(r, rr="none"):
    if "clean" not in r or "combined" not in r:
        return None
    return 0.5 * (r["clean"][rr]["all"]["mrr"] + r["combined"][rr]["all"]["mrr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    load_category_models()
    ctx = Ctx(args.split, BACKFILLED)
    index = Index(ctx)
    qsets = query_sets(ctx, index)
    have_nobg = "nobg" in index.mats and all("nobg" in f for _, f in qsets.get("combined", [(None, {})])[:1])
    probs_cache = {}
    t0 = time.time()

    results = {}
    def run(name, spec, agg="max"):
        r = evaluate(index, ctx, qsets, spec, agg, probs_cache)
        results[name] = {"spec": spec, "agg": agg, "res": r, "primary": primary(r), "primary_fused": primary(r, "fused")}
        print(f"  {name:28s} primary={results[name]['primary'] or 0:.4f} fused={results[name]['primary_fused'] or 0:.4f} "
              f"R@1 clean={r['clean']['none']['all']['r1']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    run("A dino", {"dino": 1.0})
    if have_nobg:
        run("B dino_nobg", {"nobg": 1.0})
    run("C clip", {"clip": 1.0})
    for w in W_GRID:
        run(f"D dino+clip w={w}", {"dino": w, "clip": 1 - w})
    best_d = max([n for n in results if n.startswith("D ")], key=lambda n: results[n]["primary"])
    if have_nobg:
        for w in W_GRID:
            run(f"B+C nobg+clip w={w}", {"nobg": w, "clip": 1 - w})
    singles = [n for n in results if not n.startswith(("D ", "B+C"))] + [best_d]
    if have_nobg:
        singles.append(max([n for n in results if n.startswith("B+C")], key=lambda n: results[n]["primary"]))
    best = max(singles, key=lambda n: results[n]["primary"])
    for agg in ["mean", "maxmean"]:
        run(f"E {best} | {agg}", results[best]["spec"], agg)
    winner = max(list(results), key=lambda n: results[n]["primary"])

    # sizes / latency
    dims = {"dino": 768, "clip": 512, "nobg": 768}
    q_lat = {"dino": 0.123, "clip": None, "nobg": None}   # dino p50 from Phase 1 (per crop, batch 1)
    from PIL import Image
    from src.eval.common import open_image
    from src.catalogue.build_clip_embeddings import clip_image_embed
    img = open_image(ctx.queries[0]["path"], 1024)
    ts = []
    for _ in range(6):
        t = time.perf_counter(); clip_image_embed([img]); ts.append(time.perf_counter() - t)
    q_lat["clip"] = float(np.median(ts[1:]))
    if have_nobg:
        from src.catalogue.build_nobg_embeddings import remove_bg
        ts = []
        for _ in range(4):
            t = time.perf_counter(); remove_bg(img); ts.append(time.perf_counter() - t)
        q_lat["nobg"] = float(np.median(ts[1:])) + q_lat["dino"]
    for n, r in results.items():
        used = [k for k, w in r["spec"].items() if w]
        r["index_mb"] = sum(dims[k] for k in used) * 4 * len(ctx.cid_of_fid) / 1e6
        r["query_embed_s"] = sum(q_lat[k] for k in used if q_lat[k] is not None)
        q = qsets["clean"][0][1]
        t = time.perf_counter()
        for _ in range(20):
            index.rank(variant_scores(index, q, r["spec"]), 0, r["agg"])
        r["search_ms"] = (time.perf_counter() - t) / 20 * 1000

    p1 = load_json("results/baseline.json")["val"]["retrieval_clean"]
    report = {"created": datetime.now().isoformat(timespec="seconds"), "split": args.split, "winner": winner,
              "have_nobg": have_nobg, "phase1_hnsw_prod": p1["prod"]["all"], "phase1_hnsw_raw": p1["raw"]["all"],
              "variants": results}
    if not args.no_log:
        save_json(report, "results/phase6.json")
        replace_log_section("## Phase 6 — embedding variants", markdown(report))
    print(markdown(report))


def markdown(rep):
    V = rep["variants"]
    L = [f"## Phase 6 — embedding variants ({rep['created']}, split = {rep['split']})", "",
         "Same-listing protocol, exact search, query image excluded. Primary = mean of MRR(clean, all 502) and "
         "MRR(combined phone-photo augmentation, 300 subset), chosen on val. *fused* = with the Phase 4 fused re-rank "
         "(production). Graded nDCG uses trusted labels only (title/variant, no image-model fills)."
         + ("" if rep["have_nobg"] else " **Variant B not yet available.**"), "",
         f"HNSW check (variant A, Phase 1 numbers): R@1 raw {rep['phase1_hnsw_raw']['r1']:.3f} vs exact "
         f"{V['A dino']['res']['clean']['none']['all']['r1']:.3f}.", "",
         "| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | "
         "R@1 combined aug | MRR combined aug | index MB | query embed s | search ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in V.items():
        c, a = r["res"]["clean"]["none"]["all"], r["res"].get("combined", {}).get("none", {}).get("all", {})
        L.append(f"| {n}{' **(winner)**' if n == rep['winner'] else ''} | {r['primary'] or 0:.4f} | {r['primary_fused'] or 0:.4f} | "
                 f"{c['r1']:.3f} | {c['r5']:.3f} | {c['r10']:.3f} | {c['mrr']:.3f} | {c['ndcg5']:.3f} | "
                 f"{c.get('graded_ndcg5_trusted', 0):.3f} | {c['cat_p5_non_set']:.3f} | {a.get('r1', 0):.3f} | {a.get('mrr', 0):.3f} | "
                 f"{r['index_mb']:.0f} | {r['query_embed_s']:.3f} | {r['search_ms']:.1f} |")
    w = V[rep["winner"]]
    ci = w["res"]["clean"]["none"]["all"]["ci"]
    L += ["", f"Winner `{rep['winner']}` clean R@1 {fmt_ci(ci['r1'])}, MRR {fmt_ci(ci['mrr'])}; "
          f"A dino R@1 {fmt_ci(V['A dino']['res']['clean']['none']['all']['ci']['r1'])}.", "",
          "**Per augmentation, MRR (no re-rank)**", "",
          "| aug | " + " | ".join(n for n in V if not n.startswith(("D ", "B+C", "E")) or n == rep["winner"]) + " |",
          "|---|" + "---|" * len([n for n in V if not n.startswith(("D ", "B+C", "E")) or n == rep["winner"]])]
    names = [n for n in V if not n.startswith(("D ", "B+C", "E")) or n == rep["winner"]]
    for aug in ALL_AUGS:
        if all(aug in V[n]["res"] for n in names):
            L.append(f"| {aug} | " + " | ".join(f"{V[n]['res'][aug]['none']['all']['mrr']:.3f}" for n in names) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
