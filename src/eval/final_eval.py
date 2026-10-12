"""
Phase 10 — final evaluation: v1 baseline vs v2 final, ablation by phase, latency per stage.

  1. Own photos: the FINAL production pipeline (config as deployed: fused category, calibrated
     thresholds + out-of-scope check, MMR off for photo search, preprocess) on every stumper photo,
     each stage timed with the same functions run_pipeline uses. Holdout reported (once) against the
     v1 baseline detections from Phase 3.
  2. Catalogue test split (same-listing): v1 baseline (Phase 1, HNSW, legacy re-rank) vs final
     (frozen DINOv2 + fused re-rank, exact search == HNSW here, Phase 6 check), every augmentation.
  3. Ablation: each phase's measured contribution, compiled from results/*.json.

Usage: python src/eval/final_eval.py -> results/phase10.json, results/LOG.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.append(".")
import config
from src.eval.common import STUMPER_SPLIT_PATH, Progress, fmt_ci, load_json, save_json, replace_log_section
from src.eval import own_eval
from src.eval.augment import ALL_AUGS
from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.detect import detect, crop_with_padding
from src.pipeline.embed import embed_image_with_raw
from src.pipeline.retrieve import search, apply_rerank
from src.pipeline.category import categorize
from src.pipeline.decision import decide
from src.pipeline.recommend import mmr, split_tiers, out_of_scope_prob
from src.pipeline.run_pipeline import load_all_models

FINAL_DETS = "data/eval_own/phase10_final_detections.json"
BASE_DETS = "data/eval_own/baseline_detections.json"


def timed_pipeline(path, lat):
    """run_pipeline() step by step, same functions, with a timer per stage."""
    t = time.perf_counter(); img = load_and_preprocess(path); lat["preprocess"].append(time.perf_counter() - t)
    t = time.perf_counter(); dets = detect(img); lat["detect"].append(time.perf_counter() - t)
    items = []
    for i, d in enumerate(dets):
        t = time.perf_counter(); crop = crop_with_padding(img, d["box"]); lat["crop"].append(time.perf_counter() - t)
        t = time.perf_counter(); q, raw = embed_image_with_raw(crop); lat["embed (DINOv2)"].append(time.perf_counter() - t)
        t = time.perf_counter(); results = search(q, top_k=20); lat["search (HNSW)"].append(time.perf_counter() - t)
        t = time.perf_counter(); fused = categorize(crop, q, raw, d["category"], results)
        lat["category (kNN + CLIP + clf)"].append(time.perf_counter() - t)
        t = time.perf_counter(); p_oos = out_of_scope_prob(fused["clip_emb"]); lat["out-of-scope check"].append(time.perf_counter() - t)
        t = time.perf_counter(); results = apply_rerank(results, fused["probs"]); lat["re-rank"].append(time.perf_counter() - t)
        t = time.perf_counter(); dec = decide(results, p_oos=p_oos); lat["decide"].append(time.perf_counter() - t)
        t = time.perf_counter(); results = mmr(results); closest, similar = split_tiers(results); lat["MMR + tiers"].append(time.perf_counter() - t)
        items.append({"item_index": i, "box": [float(v) for v in d["box"]], "detector_category": d["category"],
                      "detector_score": float(d["score"]), "classifier_category": fused["category"],
                      "classifier_confidence": fused["confidence"], "status": dec["status"], "reason": dec["reason"],
                      "p_oos": p_oos, "similarity": float(dec["similarity"]), "n_closest": len(closest),
                      "top5": [{"catalogue_id": r["catalogue_id"], "category": r["category"], "image_path": r["image_path"],
                                "score": float(r["score"])} for r in results[:5]]})
    return img.size, items


def photo_rates(dets, split, half):
    P = [v for p, v in dets.items() if split[p]["half"] == half and v["folder"] != "negatives" and v["items"]]
    N = [(p, v) for p, v in dets.items() if split[p]["half"] == half and v["folder"] == "negatives"]
    from src.eval.common import bootstrap_ci
    return {"frr": bootstrap_ci([float(all(i["status"] == "no_match" for i in v["items"])) for v in P]),
            "far": bootstrap_ci([float(any(i["status"] != "no_match" for i in v["items"])) for _, v in N]),
            "far_oos": bootstrap_ci([float(any(i["status"] != "no_match" for i in v["items"])) for p, v in N
                                     if own_eval.negative_type_guess(p) == "out_of_scope_jewellery"]),
            "far_nonjewel": bootstrap_ci([float(any(i["status"] != "no_match" for i in v["items"])) for p, v in N
                                          if own_eval.negative_type_guess(p) == "non_jewellery"])}


def catalogue_test():
    from src.eval.embedding_eval import Index, query_sets, evaluate, BACKFILLED
    from src.eval.retrieval_eval import Ctx
    from src.pipeline.category import load_category_models
    load_category_models()
    ctx = Ctx("test", BACKFILLED)
    index = Index(ctx)
    r = evaluate(index, ctx, query_sets(ctx, index), {"dino": 1.0}, "max", {})
    return r


def main():
    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    load_all_models()
    lat, dets = defaultdict(list), {}
    per_photo = []
    prog = Progress(len(split), "final")
    for i, (p, info) in enumerate(split.items()):
        t0 = time.perf_counter()
        size, items = timed_pipeline(p, lat)
        per_photo.append(time.perf_counter() - t0)
        dets[p] = {"folder": info["folder"], "half": info["half"], "size": list(size), "items": items}
        prog.update(i + 1)
    save_json(dets, FINAL_DETS)
    latency = {k: {"p50_ms": float(np.percentile(v, 50) * 1000), "p95_ms": float(np.percentile(v, 95) * 1000), "n": len(v)}
               for k, v in lat.items()}
    latency["TOTAL per photo"] = {"p50_ms": float(np.percentile(per_photo, 50) * 1000),
                                  "p95_ms": float(np.percentile(per_photo, 95) * 1000), "n": len(per_photo)}

    labels, graded = own_eval.load_labels(), own_eval.load_grades()
    base = load_json(BASE_DETS)
    own = {"baseline": own_eval.evaluate(base, split, labels, graded, "holdout", "classifier_category"),
           "final": own_eval.evaluate(dets, split, labels, graded, "holdout", "classifier_category")}
    rates = {"baseline": photo_rates(base, split, "holdout"), "final": photo_rates(dets, split, "holdout")}

    test_final = catalogue_test()
    p1 = load_json("results/baseline.json")["test"]
    rep = {"created": datetime.now().isoformat(timespec="seconds"), "latency": latency, "own_holdout": own,
           "open_set_holdout": rates, "test_final": test_final,
           "test_baseline": {"clean": p1["retrieval_clean"]["prod"], "aug": p1.get("retrieval_aug", {})},
           "config": {k: getattr(config, k) for k in ("CATEGORY_MODE", "HIGH_MATCH_THRESHOLD", "LOW_MATCH_THRESHOLD",
                                                      "MARGIN_THRESHOLD", "OOS_THRESHOLD", "MMR_LAMBDA_PREDICT",
                                                      "MMR_LAMBDA_SIMILAR", "DETECTOR_MODEL", "EMBEDDER_MODEL")}}
    save_json(rep, "results/phase10.json")
    md = markdown(rep)
    replace_log_section("## Phase 10 — final evaluation", md)
    print(md)


def markdown(rep):
    ob, of = rep["own_holdout"]["baseline"], rep["own_holdout"]["final"]
    rb, rf = rep["open_set_holdout"]["baseline"], rep["open_set_holdout"]["final"]
    tb, tf = rep["test_baseline"]["clean"]["all"], rep["test_final"]["clean"]["fused"]["all"]
    L = [f"## Phase 10 — final evaluation ({rep['created']})", "",
         f"Final config: `{rep['config']}`.", "",
         "**A. Real phone photos (stumper holdout, 73 photos; 95% bootstrap CIs; never used to choose)**", "",
         "| metric | v1 baseline | v2 final |", "|---|---|---|"]
    for k, n in (("detection_recall", "Detection recall"), ("category_acc_first", "Category accuracy (first item)"),
                 ("item_category_f1", "Item-category F1"), ("count_exact", "Count exact")):
        L.append(f"| {n} | {fmt_ci(ob[k])} | {fmt_ci(of[k])} |")
    for k, n in (("ndcg5", "Graded nDCG@5"), ("p5", "P@5 (grade ≥ 2)"), ("s1", "Success@1"), ("judged", "judged@5")):
        L.append(f"| {n} | {fmt_ci(ob['graded'][k])} | {fmt_ci(of['graded'][k])} |")
    L += [f"| False-reject rate (positive photos) | {fmt_ci(rb['frr'])} | {fmt_ci(rf['frr'])} |",
          f"| False-accept rate (negative photos) | {fmt_ci(rb['far'])} | {fmt_ci(rf['far'])} |",
          f"| … non-jewellery | {fmt_ci(rb['far_nonjewel'])} | {fmt_ci(rf['far_nonjewel'])} |",
          f"| … out-of-scope jewellery | {fmt_ci(rb['far_oos'])} | {fmt_ci(rf['far_oos'])} |", "",
          "**B. Catalogue retrieval, test split (same-listing protocol)**", "",
          "| metric | v1 baseline (legacy re-rank, HNSW) | v2 final (fused re-rank) |", "|---|---|---|"]
    for k, n in (("r1", "R@1"), ("r5", "R@5"), ("r10", "R@10"), ("mrr", "MRR"), ("ndcg5", "nDCG@5"),
                 ("cat_p5_non_set", "category P@5 (non-set)"), ("graded_ndcg5_trusted", "graded nDCG@5 (trusted labels)")):
        fb = "–" if tb.get(k) is None else f"{tb[k]:.3f}"          # Phase 1 did not compute every metric
        L.append(f"| {n} | {fb} | {(tf.get(k) or 0):.3f} |")
    L += ["", "Per augmentation (test, R@1 / MRR):", "", "| aug | v1 baseline | v2 final |", "|---|---|---|"]
    for aug in ALL_AUGS:
        b = rep["test_baseline"]["aug"].get(aug, {}).get("prod", {}).get("all")
        f = rep["test_final"].get(aug, {}).get("fused", {}).get("all")
        if b and f:
            L.append(f"| {aug} | {b['r1']:.3f} / {b['mrr']:.3f} | {f['r1']:.3f} / {f['mrr']:.3f} |")
    L += ["", "**C. Latency per stage, final pipeline (CPU, one image at a time, per detected item unless noted)**", "",
          "| stage | p50 ms | p95 ms | n |", "|---|---|---|---|"]
    for k, v in rep["latency"].items():
        L.append(f"| {k} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['n']} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
