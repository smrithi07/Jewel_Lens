"""
Phase 1 — re-measure the v1 stumper numbers on the fixed tune/holdout split
(data/stumper/split.json), with photo-level bootstrap CIs. Headline = holdout.

Ground truth comes ONLY from the folder name (earring/necklace/bracelet) and
the negatives folders. labels_draft.csv is not read: its chosen_rank /
label_source=clip columns are circular. `mixed` photos count toward
Detection Recall only (their per-item categories arrive in Phase 3).

Two input paths, both through the unchanged production run_pipeline():
  as_deployed     : Image.open(...).convert("RGB") — what src/api/app.py does today
                    (no EXIF transpose, no 1600px cap) and what v1's harness did.
  with_preprocess : src/pipeline/preprocess.preprocess_image() first.

Metrics (per photo)
  detection_recall  : positive photos (incl. mixed) with >=1 detection
  category_acc_v1   : single-category photos with >=1 detection, first item's
                      classifier label == folder (v1's definition)
  detector_label_acc: same, using the first item's OWL-ViT label (what the UI shows)
  item_category_acc : fraction of all detected items whose classifier label == folder
  no_match_rate_neg : negatives where every item is no_match (or nothing detected).
                      v1 called this "no-match precision"; it is really the
                      true-negative rate, renamed here.
  false_reject_pos  : positive photos where every item came back no_match

Usage: python src/eval/stumper_baseline.py [--tag baseline]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.append(".")
import config
from src.eval.common import (STUMPER_SPLIT_PATH, RESULTS_DIR, Progress, bootstrap_ci, fmt_ci,
                             load_json, save_json, append_log)
from src.pipeline.run_pipeline import run_pipeline, load_all_models
from src.pipeline.preprocess import preprocess_image

CATS = config.CATEGORIES
MODES = ["as_deployed", "with_preprocess"]


def photo_record(path, folder, items):
    first = items[0] if items else None
    return {
        "photo": path, "folder": folder, "n_items": len(items),
        "first_classifier": first["classifier_category"] if first else None,
        "first_detector": first["detector_category"] if first else None,
        "item_classifier": [i["classifier_category"] for i in items],
        "item_detector": [i["detector_category"] for i in items],
        "statuses": [i["decision"]["status"] for i in items],
        "top1_sim": [float(i["decision"]["similarity"]) for i in items],
    }


def summarise(records):
    pos = [r for r in records if r["folder"] in CATS + ["mixed"]]
    single = [r for r in records if r["folder"] in CATS and r["n_items"] > 0]
    neg = [r for r in records if r["folder"] == "negatives"]
    items = [(r["folder"], c) for r in single for c in r["item_classifier"]]
    out = {
        "n_photos": len(records), "n_positive": len(pos), "n_single_detected": len(single), "n_negative": len(neg),
        "detection_recall": bootstrap_ci([float(r["n_items"] > 0) for r in pos]),
        "category_acc_v1": bootstrap_ci([float(r["first_classifier"] == r["folder"]) for r in single]),
        "detector_label_acc": bootstrap_ci([float(r["first_detector"] == r["folder"]) for r in single]),
        "item_category_acc": bootstrap_ci([float(np.mean([c == r["folder"] for c in r["item_classifier"]])) for r in single]),
        "no_match_rate_neg": bootstrap_ci([float(all(s == "no_match" for s in r["statuses"])) for r in neg]),
        "false_reject_pos": bootstrap_ci([float(r["n_items"] > 0 and all(s == "no_match" for s in r["statuses"])) for r in pos]),
        "mean_items_per_positive": float(np.mean([r["n_items"] for r in pos])) if pos else None,
        "confusion_classifier": {t: dict(Counter(r["first_classifier"] for r in single if r["folder"] == t)) for t in CATS},
        "confusion_detector": {t: dict(Counter(r["first_detector"] for r in single if r["folder"] == t)) for t in CATS},
        "n_items_scored": len(items),
    }
    out["per_category_acc_v1"] = {t: bootstrap_ci([float(r["first_classifier"] == t) for r in single if r["folder"] == t])
                                  for t in CATS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    load_all_models()
    t_start = time.time()

    records = {m: [] for m in MODES}
    latency = []
    prog = Progress(len(split) * len(MODES), "stumper")
    done = 0
    for path, info in split.items():
        base = Image.open(path)
        for mode in MODES:
            img = base.convert("RGB") if mode == "as_deployed" else preprocess_image(base)
            t0 = time.perf_counter()
            items = run_pipeline(img, top_k=5)
            latency.append((mode, time.perf_counter() - t0))
            rec = photo_record(path, info["folder"], items)
            rec["half"] = info["half"]
            records[mode].append(rec)
            done += 1
            prog.update(done)

    res = {"created": datetime.now().isoformat(timespec="seconds"),
           "notes": {"ground_truth": "folder name only; labels_draft.csv not used (circular)",
                     "mixed": "detection recall only until Phase 3 item labels",
                     "no_match_rate_neg": "v1 called this 'no-match precision'; it is the true-negative rate",
                     "headline": "holdout"},
           "latency_ms": {m: {"p50": float(np.percentile([t for mm, t in latency if mm == m], 50) * 1000),
                              "p95": float(np.percentile([t for mm, t in latency if mm == m], 95) * 1000)} for m in MODES},
           "records": records}
    for mode in MODES:
        res[mode] = {half: summarise([r for r in records[mode] if half == "all" or r["half"] == half])
                     for half in ("holdout", "tune", "all")}

    out_path = os.path.join(RESULTS_DIR, f"{args.tag}_stumper.json")
    save_json(res, out_path)

    lines = [f"## Phase 1 — {args.tag}, stumper photos ({res['created']})", "",
             "Ground truth = folder name. `mixed` counts toward detection recall only. 95% bootstrap CIs by photo. "
             "**Headline = holdout.** `as_deployed` = what the API does today (no EXIF/resize); "
             "`with_preprocess` = preprocess_image() first.", "",
             "| input | half | photos | Det. recall | Category acc (v1 def.) | Detector-label acc | Item category acc | Neg. rejected | Pos. false-reject |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode in MODES:
        for half in ("holdout", "tune", "all"):
            s = res[mode][half]
            lines.append(f"| {mode} | {half} | {s['n_photos']} | {fmt_ci(s['detection_recall'])} | {fmt_ci(s['category_acc_v1'])} | "
                         f"{fmt_ci(s['detector_label_acc'])} | {fmt_ci(s['item_category_acc'])} | "
                         f"{fmt_ci(s['no_match_rate_neg'])} (n={s['n_negative']}) | {fmt_ci(s['false_reject_pos'])} |")
    h = res["as_deployed"]["holdout"]
    lines += ["", "Holdout confusion, as_deployed (rows = true folder, first item):", "",
              "| true | classifier → " + " / ".join(CATS) + " | detector → " + " / ".join(CATS) + " |", "|---|---|---|"]
    for t in CATS:
        lines.append(f"| {t} | " + " / ".join(str(h['confusion_classifier'][t].get(c, 0)) for c in CATS)
                     + " | " + " / ".join(str(h['confusion_detector'][t].get(c, 0)) for c in CATS) + " |")
    lines += ["", f"Pipeline latency per photo (ms): " + ", ".join(
        f"{m} p50 {v['p50']:.0f} / p95 {v['p95']:.0f}" for m, v in res["latency_ms"].items())]
    if not args.no_log:
        append_log("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_path} in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
