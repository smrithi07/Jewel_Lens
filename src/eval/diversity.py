"""
Phase 8 — MMR diversity re-rank, lambda chosen on catalogue val.

Production ranking (frozen DINOv2, exact top-20 listings, Phase 4 fused re-rank), then
Maximal Marginal Relevance over those 20:
    next = argmax_i  lambda * score(i) - (1 - lambda) * max_{j in chosen} cos(img_i, img_j)
where img_i is the listing image that matched the query. lambda = 1 is the current ranking.

Reported per lambda: R@5, nDCG@5, trusted graded nDCG@5, category P@5, intra-list diversity
ILD@5 = 1 - mean pairwise cosine of the top-5 matched images, and the share of top-5 pairs that
are visual near-duplicates (cosine >= 0.9). Rule (fixed in advance): the most diverse lambda whose
nDCG@5 and R@5 both stay >= 98% of lambda = 1, on clean AND combined-augmentation queries.

Usage: python src/eval/diversity.py -> results/phase8_mmr.json, LOG.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from datetime import datetime

import numpy as np

sys.path.append(".")
from src.eval.common import save_json, replace_log_section
from src.eval.embedding_eval import Index, query_sets, fused_probs, BACKFILLED
from src.eval.retrieval_eval import Ctx, score, aggregate
from src.pipeline.retrieve import apply_rerank
from src.pipeline.category import load_category_models

LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
KEEP = 0.98


def mmr(ranked, emb, lam, k=None):
    """ranked: list of dicts with 'score' and 'fid' (matched image); emb: normalised image matrix."""
    k = k or len(ranked)
    cand = list(range(len(ranked)))
    V = emb[[r["fid"] for r in ranked]]
    sims = V @ V.T
    chosen = []
    while cand and len(chosen) < k:
        if not chosen:
            best = max(cand, key=lambda i: ranked[i]["score"])
        else:
            best = max(cand, key=lambda i: lam * ranked[i]["score"] - (1 - lam) * max(sims[i, j] for j in chosen))
        chosen.append(best)
        cand.remove(best)
    return [ranked[i] for i in chosen]


def ild(results, emb, k=5):
    V = emb[[r["fid"] for r in results[:k]]]
    S = V @ V.T
    iu = np.triu_indices(len(V), 1)
    return float(1 - S[iu].mean()), float((S[iu] >= 0.9).mean())


def main():
    load_category_models()
    ctx = Ctx("val", BACKFILLED)
    index = Index(ctx)
    emb = index.mats["dino"]
    qsets = query_sets(ctx, index)
    raw_clean = qsets["_clean_raw"]
    pos = {c: i for i, c in enumerate(index.cid)}

    def ranked_with_fids(q, f, raw):
        sims = emb @ f["dino"]
        ranked = index.rank(sims, q["fid"], "max")
        for r in ranked:
            li = pos[r["catalogue_id"]]
            a, n = index.starts[li], index.counts[li]
            seg = sims[a:a + n].copy()
            if a <= q["fid"] < a + n:
                seg[q["fid"] - a] = -np.inf
            r["fid"] = int(a + np.argmax(seg))
        return apply_rerank([dict(r) for r in ranked], fused_probs(index, ctx, q, f, raw))

    base = {}
    for aug in ("clean", "combined"):
        base[aug] = [(q, ranked_with_fids(q, f, raw_clean[q["cid"]] if aug == "clean" else f["dino_raw"]))
                     for q, f in qsets[aug]]

    res = {}
    for lam in LAMBDAS:
        res[lam] = {}
        for aug, rows in base.items():
            sc, div, dup = [], [], []
            for q, rr in rows:
                out = rr if lam == 1.0 else mmr(rr, emb, lam)
                sc.append((q, score(ctx, q, out)))
                d, nd = ild(out, emb)
                div.append(d)
                dup.append(nd)
            a = aggregate(sc)["all"]
            res[lam][aug] = {"r5": a["r5"], "ndcg5": a["ndcg5"], "graded_trusted": a.get("graded_ndcg5_trusted"),
                             "cat_p5": a["cat_p5_non_set"], "ild5": float(np.mean(div)), "near_dup_pairs": float(np.mean(dup))}
        print(f"  lambda={lam}: " + " | ".join(f"{aug} nDCG@5 {v['ndcg5']:.3f} R@5 {v['r5']:.3f} ILD {v['ild5']:.3f} dup {v['near_dup_pairs']:.3f}"
                                              for aug, v in res[lam].items()), flush=True)

    ok = [l for l in LAMBDAS if all(res[l][a]["ndcg5"] >= KEEP * res[1.0][a]["ndcg5"] and res[l][a]["r5"] >= KEEP * res[1.0][a]["r5"]
                                    for a in base)]
    chosen = min(ok) if ok else 1.0
    rep = {"created": datetime.now().isoformat(timespec="seconds"), "lambda": chosen, "rule": f"most diverse with nDCG@5 and R@5 >= {KEEP:.0%} of lambda=1",
           "results": {str(k): v for k, v in res.items()}}
    save_json(rep, "results/phase8_mmr.json")
    L = [f"## Phase 8 — MMR diversity ({rep['created']})", "",
         f"Catalogue val, production ranking (frozen DINOv2 + fused re-rank), MMR over the top 20. Rule: {rep['rule']} on clean and "
         f"combined-aug queries. **Chosen λ = {chosen}.**", "",
         "| λ | clean nDCG@5 | clean R@5 | clean graded (trusted) | clean cat P@5 | clean ILD@5 | clean near-dup pairs | combined nDCG@5 | combined R@5 | combined ILD@5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for l in LAMBDAS:
        c, m = res[l]["clean"], res[l]["combined"]
        L.append(f"| {l}{' **(chosen)**' if l == chosen else ''} | {c['ndcg5']:.3f} | {c['r5']:.3f} | {c['graded_trusted']:.3f} | {c['cat_p5']:.3f} | "
                 f"{c['ild5']:.3f} | {c['near_dup_pairs']:.3f} | {m['ndcg5']:.3f} | {m['r5']:.3f} | {m['ild5']:.3f} |")
    replace_log_section("## Phase 8 — MMR diversity", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
