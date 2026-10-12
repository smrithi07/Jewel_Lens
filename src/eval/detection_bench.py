"""
Phase 5 — detector benchmark, prompt ensembling and post-processing ablation.
Everything is CHOSEN ON TUNE; holdout is reported once for the final config.

Steps
  1. detector benchmark: owlvit / owlv2 / gdino with the v1 prompts; score threshold and
     NMS IoU tuned per detector on tune (objective below)
  2. prompt ensembling on the chosen detector: current vs ensemble (max / mean of
     phrasings) x generic "jewellery" phrases on/off, threshold re-tuned
  3. post-processing rules on top: pair_protect, group_drop, chain_merge, min box size,
     nms_per_class, SAHI-style tiles, retrieval verification — each added alone, then a joint
     grid over the interacting ones (tiles x verification x group_drop)
  Anti-overfitting: every adoption (tuned threshold, other detector, prompt ensemble, rules)
  must beat the current choice by > MARGIN = 0.02 tune F1; otherwise the simpler config stays
  4. holdout report: baseline (v1 detector + Phase 4 fused category) vs final, per tag

Objective on tune (per photo, mean): item-category F1 against your labelled per-category
counts (it rewards the right number of items AND the right category), tie-break count
exact, then lower latency. Constraint: detection recall >= baseline recall - 0.02.
Detector choice also needs a real gain: a slower detector must beat the fastest by
> 0.02 tune F1.

Category of every box = Phase 4 fused category (kNN + CLIP + clf_v2), computed from a
crop-feature cache, so the comparison isolates detection.

Usage: python src/eval/detection_bench.py   -> results/phase5.json, LOG.md,
       data/eval_own/phase5_final_detections.json, artifacts/detection_v2/config.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import sys
import time
from datetime import datetime

import numpy as np

sys.path.append(".")
import config
from src.eval.common import STUMPER_SPLIT_PATH, Progress, fmt_ci, load_json, save_json, replace_log_section
from src.eval import own_eval
from src.pipeline.postprocess import postprocess, DEFAULT
from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.detect import crop_with_padding
from src.pipeline.retrieve import search, apply_rerank
from src.pipeline.decision import decide
from src.pipeline.category import categorize, load_category_models

DETECTORS = ["owlvit", "owlv2", "gdino"]
THR_GRID = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
NMS_GRID = [0.3, 0.5]
CROP_CACHE = "data/eval_own/phase5_crop_cache.npz"
MARGIN = 0.02   # a change must beat the current choice by > this tune F1 (~1 photo) to be adopted
TAGS_REPORT = ["chain", "overlap", "hand_held", "pendant", "multi_item", "no_tag"]


# ------------------------------------------------------------------ crop features (cached by box)

class CropCache:
    def __init__(self):
        self.d = {}
        if os.path.exists(CROP_CACHE):
            z = np.load(CROP_CACHE, allow_pickle=True)
            self.d = z["d"].item()
        self.dirty = 0
        self._imgs = {}

    def key(self, photo, box):
        return f"{photo}|" + ",".join(f"{v:.0f}" for v in box)

    def get(self, photo, box):
        k = self.key(photo, box)
        if k not in self.d:
            from src.eval.retrieval_eval import embed_batch_raw
            from src.catalogue.build_clip_embeddings import clip_image_embed
            if photo not in self._imgs:
                self._imgs = {photo: load_and_preprocess(photo)}
            crop = crop_with_padding(self._imgs[photo], box)
            raw = embed_batch_raw([crop])[0].astype(np.float32)
            norm = raw / np.linalg.norm(raw)
            res = search(norm, top_k=20)
            self.d[k] = {"raw": raw, "clip": clip_image_embed([crop])[0].astype(np.float32),
                         "res": [{"catalogue_id": r["catalogue_id"], "category": r["category"],
                                  "image_path": r["image_path"], "score": float(r["score"])} for r in res]}
            self.dirty += 1
            if self.dirty % 50 == 0:
                self.save()
        return self.d[k]

    def save(self):
        np.savez_compressed(CROP_CACHE, d=np.array(self.d, dtype=object))


# ------------------------------------------------------------------ build a detections file for a config

def raw_for(det_name, cache, photo, cfg):
    """-> (dets, queries, size, seconds, tile_dets). Tile latency is added only when tiles are used."""
    e = cache[photo]
    if det_name == "gdino":
        ps = e["passes"]["ensemble" if cfg["query_set"] == "ensemble" else "current"]
    else:
        ps = e["passes"]["all"]
    tiles = e["passes"].get("tiles") if cfg.get("tiles") else None
    sec = ps["seconds"] + (tiles["seconds"] if tiles else 0.0)
    return ps["dets"], ps["queries"], e["size"], sec, (tiles["dets"] if tiles else None)


def count_only(det_name, cache, photos, cfg):
    """Fast path: boxes + detector labels only (no crop features)."""
    out = {}
    for p in photos:
        dets, queries, size, sec, tdets = raw_for(det_name, cache, p, cfg)
        out[p] = postprocess(dets, queries, size, cfg, tile_raw=tdets)
    return out


def full_detections(det_name, cache, photos, cfg, split, crops, with_retrieval=False):
    dets_out, lat = {}, []
    for p in photos:
        dets, queries, size, sec, tdets = raw_for(det_name, cache, p, cfg)
        t0 = time.perf_counter()
        items = postprocess(dets, queries, size, cfg, tile_raw=tdets)
        lat.append(sec + time.perf_counter() - t0)
        if cfg.get("verify_sim") is not None and items:
            # retrieval verification: a box must look like *some* catalogue item; keep the top box regardless
            top = max(items, key=lambda x: x["score"])
            items = [it for it in items if it is top or crops.get(p, it["box"])["res"][0]["score"] >= cfg["verify_sim"]]
        out_items = []
        for i, it in enumerate(items):
            f = crops.get(p, it["box"])
            norm = f["raw"] / np.linalg.norm(f["raw"])
            fused = categorize(None, norm, f["raw"], it["detector_category"], f["res"], clip_emb=f["clip"])
            rec = {"item_index": i, "box": it["box"], "detector_category": it["detector_category"],
                   "detector_score": it["score"], "fused_category": fused["category"],
                   "classifier_category": fused["category"], "classifier_confidence": fused["confidence"]}
            if with_retrieval:
                ranked = apply_rerank([dict(r) for r in f["res"]], fused["probs"])
                rec["top5"] = ranked[:5]
                rec["status"] = decide(ranked)["status"]
            else:
                rec["top5"], rec["status"] = [], "match"
            out_items.append(rec)
        dets_out[p] = {"folder": split[p]["folder"], "half": split[p]["half"], "size": size, "items": out_items}
    return dets_out, lat


def metrics(dets, split, labels, graded, half):
    r = own_eval.evaluate(dets, split, labels, graded, half, "fused_category")
    return r


def score_key(r, lat):
    return (round(r["item_category_f1"]["mean"] or 0, 4), round(r["count_exact"]["mean"] or 0, 4), -np.median(lat))


# ------------------------------------------------------------------ main

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    dets_wanted = args.detectors.split(",")
    load_category_models()
    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    labels = own_eval.load_labels()
    graded = own_eval.load_grades()
    tune = [p for p, v in split.items() if v["half"] == "tune"]
    hold = [p for p, v in split.items() if v["half"] == "holdout"]
    caches = {d: load_json(f"data/eval_own/phase5_raw_{d}.json") for d in dets_wanted
              if os.path.exists(f"data/eval_own/phase5_raw_{d}.json")}
    caches = {d: c for d, c in caches.items() if all(p in c for p in split)}   # skip partial caches
    missing = [d for d in DETECTORS if d not in caches]
    crops = CropCache()
    report = {"created": datetime.now().isoformat(timespec="seconds"), "missing_detectors": missing}

    # baseline: v1 detector config + fused category
    base_cfg = dict(DEFAULT)
    bdets, blat = full_detections("owlvit", caches["owlvit"], tune, base_cfg, split, crops)
    base_tune = metrics(bdets, split, labels, [], "tune")
    recall_floor = base_tune["detection_recall"]["mean"] - 0.02
    report["baseline_tune"] = summarize(base_tune, blat)

    def tune_cfgs(det, cfg_grid, label):
        """Count-only screen of every cfg, then full (category) eval of the top 5 by count metrics."""
        screened = []
        for cfg in cfg_grid:
            dd = count_only(det, caches[det], tune, cfg)
            dd = {p: {"folder": split[p]["folder"], "half": "tune", "items":
                      [{**it, "detector_score": it["score"], "fused_category": it["detector_category"], "status": "match", "top5": []}
                       for it in v]}
                  for p, v in dd.items()}
            r = metrics(dd, split, labels, [], "tune")
            if r["detection_recall"]["mean"] >= recall_floor:
                screened.append((r["count_exact"]["mean"] or 0, -(r["count_mae"]["mean"] or 9), cfg))
        screened.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = None
        prog = Progress(min(5, len(screened)), label)
        for i, (_, _, cfg) in enumerate(screened[:5]):
            dd, lat = full_detections(det, caches[det], tune, cfg, split, crops)
            r = metrics(dd, split, labels, [], "tune")
            if best is None or score_key(r, lat) > score_key(best[1], best[2]):
                best = (cfg, r, lat)
            prog.update(i + 1)
        crops.save()
        return best

    # ---- 1. detector benchmark (v1 prompts)
    bench = {}
    for det in caches:
        grid = [{**DEFAULT, "score_thr": t, "nms_iou": n} for t in THR_GRID for n in NMS_GRID]
        cfg, r, lat = tune_cfgs(det, grid, f"bench:{det}")
        bench[det] = {"cfg": cfg, "tune": summarize(r, lat)}
        print(f"  {det:7s} thr={cfg['score_thr']} nms={cfg['nms_iou']} F1={r['item_category_f1']['mean']:.3f} "
              f"count={r['count_exact']['mean']:.3f} recall={r['detection_recall']['mean']:.3f} lat={np.median(lat):.2f}s")
    report["detector_benchmark"] = bench
    fastest = min(bench, key=lambda d: bench[d]["tune"]["latency_p50_s"])
    chosen = fastest
    for d in bench:
        if bench[d]["tune"]["item_category_f1"] > bench[chosen]["tune"]["item_category_f1"] + MARGIN:
            chosen = d
    report["chosen_detector"] = chosen
    # the tuned threshold/NMS must beat the v1 config by the margin, else keep v1's (owlvit only)
    if chosen == "owlvit" and bench["owlvit"]["tune"]["item_category_f1"] <= report["baseline_tune"]["item_category_f1"] + MARGIN:
        bench["owlvit"]["rejected_tuned"] = {"cfg": bench["owlvit"]["cfg"], "tune": bench["owlvit"]["tune"]}
        bench["owlvit"]["cfg"] = dict(DEFAULT)
        bench["owlvit"]["tune"] = report["baseline_tune"]
        report["owlvit_threshold_note"] = "tuned thr/NMS did not beat v1 (0.1 / 0.3) by the margin; v1 kept"

    # ---- 2. prompt ensembling on the chosen detector
    variants = [("current", "max", False), ("ensemble", "max", False), ("ensemble", "mean", False),
                ("ensemble", "max", True), ("ensemble", "mean", True)]
    prompt_res = {}
    best_prompt = None
    cur_cfg = bench[chosen]["cfg"]
    # "current" keeps the detector's chosen config; each ensemble variant gets its own threshold/NMS
    # and is adopted only if it beats "current" by the margin (and beats the other variants)
    current_f1 = None
    for qs, agg, agn in variants:
        name = f"{qs}/{agg}{'/+jewellery' if agn else ''}"
        if qs == "current":
            cfg = dict(cur_cfg)
            dd, lat = full_detections(chosen, caches[chosen], tune, cfg, split, crops)
            r = metrics(dd, split, labels, [], "tune")
            current_f1 = r["item_category_f1"]["mean"] or 0
            best_prompt = (cfg, r, lat, name)
        else:
            grid = [{**DEFAULT, "query_set": qs, "agg": agg, "agnostic_queries": agn, "score_thr": t, "nms_iou": n}
                    for t in THR_GRID for n in NMS_GRID]
            cfg, r, lat = tune_cfgs(chosen, grid, f"prompts:{name}")
            f = r["item_category_f1"]["mean"] or 0
            if f > current_f1 + MARGIN and f > (best_prompt[1]["item_category_f1"]["mean"] or 0):
                best_prompt = (cfg, r, lat, name)
        prompt_res[name] = {"cfg": cfg, "tune": summarize(r, lat)}
    report["prompt_ensembling"] = prompt_res
    report["chosen_prompts"] = best_prompt[3]

    # ---- 3. post-processing rules (each alone on top, then leave-one-out of the kept set)
    base = best_prompt[0]
    rules = {"pair_protect": {"pair_protect": True}, "group_drop": {"group_drop": True},
             "chain_merge": {"chain_merge": True}, "min_box": {"min_area_frac": 0.002},
             "nms_per_class": {"nms_per_class": True}, "tiles": {"tiles": True},
             "tiles(thr 0.15)": {"tiles": True, "tile_score_thr": 0.15},
             "tiles(thr 0.2)": {"tiles": True, "tile_score_thr": 0.2},
             "verify(0.45)": {"verify_sim": 0.45}, "verify(0.55)": {"verify_sim": 0.55}}
    if chosen != "owlvit" or "tiles" not in next(iter(caches[chosen].values()))["passes"]:
        rules = {k: v for k, v in rules.items() if not k.startswith("tiles")}

    def run(cfg):
        dd, lat = full_detections(chosen, caches[chosen], tune, cfg, split, crops)
        return metrics(dd, split, labels, [], "tune"), lat

    r0, l0 = run(base)
    f0 = r0["item_category_f1"]["mean"] or 0
    ablation = {"(none)": summarize(r0, l0)}
    singles = {}
    for name, delta in rules.items():
        r, lat = run({**base, **delta})
        ablation[f"+{name}"] = summarize(r, lat)
        singles[name] = (r["item_category_f1"]["mean"] or 0, delta)
    # joint grid over the interacting rules: tiles x verification x group_drop (all toggles of the useful ones)
    tile_opts = [{}] + [d for n, d in rules.items() if n.startswith("tiles")]
    combos = [{**t, **v, **g} for t in tile_opts for v in ({}, {"verify_sim": 0.45}, {"verify_sim": 0.55})
              for g in ({}, {"group_drop": True})]
    best = (f0, {}, r0, l0)
    for delta in combos:
        if not delta:
            continue
        r, lat = run({**base, **delta})
        f = r["item_category_f1"]["mean"] or 0
        ablation["grid " + json_name(delta)] = summarize(r, lat)
        if f > best[0]:
            best = (f, delta, r, lat)
    # adopt only with a real margin over the base config
    chosen_delta = best[1] if best[0] > f0 + MARGIN else {}
    final_cfg = {**base, **chosen_delta}
    rf, lf = run(final_cfg)
    ablation["FINAL " + (json_name(chosen_delta) or "(base, no rule beat the margin)")] = summarize(rf, lf)
    for k in chosen_delta:
        cfg = {kk: vv for kk, vv in final_cfg.items() if kk != k}
        cfg[k] = DEFAULT.get(k)
        r, lat = run(cfg)
        ablation[f"FINAL - {k}"] = summarize(r, lat)
    crops.save()
    report["rule_ablation_tune"] = ablation
    report["final_config"] = {"detector": chosen, **final_cfg}

    # ---- 4. holdout: baseline vs final, with graded retrieval
    hb, hbl = full_detections("owlvit", caches["owlvit"], hold, base_cfg, split, crops, with_retrieval=True)
    hf, hfl = full_detections(chosen, caches[chosen], hold, final_cfg, split, crops, with_retrieval=True)
    crops.save()
    save_json(hf, "data/eval_own/phase5_final_detections.json")
    save_json(hb, "data/eval_own/phase5_baseline_fused_detections.json")
    report["holdout"] = {"baseline": summarize(metrics(hb, split, labels, graded, "holdout"), hbl, full=True),
                         "final": summarize(metrics(hf, split, labels, graded, "holdout"), hfl, full=True)}
    os.makedirs("artifacts/detection_v2", exist_ok=True)
    save_json(report["final_config"], "artifacts/detection_v2/config.json")
    md = markdown(report)
    if not args.no_log:
        save_json(report, "results/phase5.json")
        replace_log_section("## Phase 5 — detection and multi-item", md)
    print(md)


def json_name(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


def summarize(r, lat, full=False):
    s = {"detection_recall": r["detection_recall"]["mean"], "count_exact": r["count_exact"]["mean"],
         "count_mae": r["count_mae"]["mean"], "item_category_f1": r["item_category_f1"]["mean"],
         "category_acc_first": r["category_acc_first"]["mean"],
         "under": r["under_count_rate"], "over": r["over_count_rate"],
         "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
         "by_tag": {t: {"n": v["n"], "count_exact": v["count_exact"]["mean"], "f1": v["item_category_f1"]["mean"]}
                    for t, v in r["by_tag"].items()}}
    if full:
        s["ci"] = {k: r[k] for k in ("detection_recall", "count_exact", "item_category_f1", "category_acc_first")}
        s["graded"] = r["graded"]
        s["negatives_rejected"] = r["negatives_rejected"]
    return s


def markdown(rep):
    L = [f"## Phase 5 — detection and multi-item ({rep['created']})", "",
         "Category of every box = Phase 4 fused category. Chosen on **tune**; holdout reported once. "
         "Objective: item-category F1 vs labelled counts, then count exact, then latency; detection recall must stay within 0.02 of baseline.",
         ""]
    if rep["missing_detectors"]:
        L += [f"Detectors not available: {rep['missing_detectors']}", ""]
    L += ["**1. Detector benchmark (tune, v1 prompts, threshold/NMS tuned per detector)**", "",
          "| detector | thr / NMS IoU | det. recall | count exact | count MAE | item-cat F1 | under / over | latency p50 / p95 (s) |",
          "|---|---|---|---|---|---|---|---|"]
    b = rep["baseline_tune"]
    L.append(f"| owlvit v1 config (baseline) | 0.1 / 0.3 | {b['detection_recall']:.3f} | {b['count_exact']:.3f} | {b['count_mae']:.2f} | "
             f"{b['item_category_f1']:.3f} | {b['under']:.2f} / {b['over']:.2f} | {b['latency_p50_s']:.2f} / {b['latency_p95_s']:.2f} |")
    for d, v in rep["detector_benchmark"].items():
        t = v["tune"]
        L.append(f"| {d}{' **(chosen)**' if d == rep['chosen_detector'] else ''} | {v['cfg']['score_thr']} / {v['cfg']['nms_iou']} | "
                 f"{t['detection_recall']:.3f} | {t['count_exact']:.3f} | {t['count_mae']:.2f} | {t['item_category_f1']:.3f} | "
                 f"{t['under']:.2f} / {t['over']:.2f} | {t['latency_p50_s']:.2f} / {t['latency_p95_s']:.2f} |")
    rj = rep["detector_benchmark"].get("owlvit", {}).get("rejected_tuned")
    if rj:
        t = rj["tune"]
        L.append(f"| owlvit tuned (rejected: < margin) | {rj['cfg']['score_thr']} / {rj['cfg']['nms_iou']} | {t['detection_recall']:.3f} | "
                 f"{t['count_exact']:.3f} | {t['count_mae']:.2f} | {t['item_category_f1']:.3f} | {t['under']:.2f} / {t['over']:.2f} | "
                 f"{t['latency_p50_s']:.2f} / {t['latency_p95_s']:.2f} |")
    L += ["", f"**2. Prompt ensembling on {rep['chosen_detector']} (tune)**", "",
          "| prompts | thr / NMS | count exact | item-cat F1 | det. recall |", "|---|---|---|---|---|"]
    for n, v in rep["prompt_ensembling"].items():
        t = v["tune"]
        L.append(f"| {n}{' **(chosen)**' if n == rep['chosen_prompts'] else ''} | {v['cfg']['score_thr']} / {v['cfg']['nms_iou']} | "
                 f"{t['count_exact']:.3f} | {t['item_category_f1']:.3f} | {t['detection_recall']:.3f} |")
    L += ["", "**3. Post-processing rule ablation (tune)**", "",
          "| config | count exact | count MAE | item-cat F1 | under / over | overlap count exact | chain count exact |",
          "|---|---|---|---|---|---|---|"]
    for n, t in rep["rule_ablation_tune"].items():
        ov, ch = t["by_tag"].get("overlap", {}), t["by_tag"].get("chain", {})
        L.append(f"| {n} | {t['count_exact']:.3f} | {t['count_mae']:.2f} | {t['item_category_f1']:.3f} | {t['under']:.2f} / {t['over']:.2f} | "
                 f"{(ov.get('count_exact') or 0):.2f} (n={ov.get('n', 0)}) | {(ch.get('count_exact') or 0):.2f} (n={ch.get('n', 0)}) |")
    L += ["", f"Final config: `{rep['final_config']}`", "", "**4. Holdout: baseline (v1 detector) vs final — both with the Phase 4 fused category**", "",
          "| metric | baseline | final |", "|---|---|---|"]
    hb, hf = rep["holdout"]["baseline"], rep["holdout"]["final"]
    for k, name in (("detection_recall", "Detection recall"), ("count_exact", "Count exact"),
                    ("item_category_f1", "Item-category F1"), ("category_acc_first", "Category acc (first item)")):
        L.append(f"| {name} | {fmt_ci(hb['ci'][k])} | {fmt_ci(hf['ci'][k])} |")
    L.append(f"| Count MAE / under / over | {hb['count_mae']:.2f} / {hb['under']:.2f} / {hb['over']:.2f} | "
             f"{hf['count_mae']:.2f} / {hf['under']:.2f} / {hf['over']:.2f} |")
    for k, name in (("ndcg5", "Graded nDCG@5"), ("p5", "P@5"), ("s1", "Success@1"), ("judged", "judged@5")):
        L.append(f"| {name} | {fmt_ci(hb['graded'][k])} | {fmt_ci(hf['graded'][k])} |")
    L.append(f"| Graded items unmatched (no box IoU ≥ 0.5) | {hb['graded']['n_unmatched']} | {hf['graded']['n_unmatched']} |")
    L.append(f"| Negatives rejected | {fmt_ci(hb['negatives_rejected'])} | {fmt_ci(hf['negatives_rejected'])} |")
    L.append(f"| Latency p50 / p95 (detect + post-process, s) | {hb['latency_p50_s']:.2f} / {hb['latency_p95_s']:.2f} | "
             f"{hf['latency_p50_s']:.2f} / {hf['latency_p95_s']:.2f} |")
    L += ["", "Per-tag holdout (count exact / item-category F1):", "", "| tag | n | baseline | final |", "|---|---|---|---|"]
    for t in TAGS_REPORT:
        a, c = hb["by_tag"].get(t), hf["by_tag"].get(t)
        if a:
            L.append(f"| {t} | {a['n']} | {(a['count_exact'] or 0):.2f} / {(a['f1'] or 0):.2f} | "
                     f"{((c or {}).get('count_exact') or 0):.2f} / {((c or {}).get('f1') or 0):.2f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
