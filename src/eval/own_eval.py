"""
Phase 3 — own-photo evaluation (stumper photos, real phone conditions).

Evaluates a DETECTIONS FILE: per photo, the items a pipeline variant produced
(box, labels, decision status, top-5). The current production pipeline's file is
data/eval_own/baseline_detections.json (made by prepare_own_eval.py, or --run).
Later phases write their own file in the same schema and point --detections at it,
so every variant is scored by exactly the same code.

Ground truth
  labels.csv  (verified=1 rows only): per-photo counts per category, failure tags,
              negative type. multi_item is added automatically when the total is >= 2.
  grades.csv  0-3 grades for the current top-5 of ~50 holdout items. A variant's item
              is matched to a graded item by IoU >= 0.5 on the same photo; results that
              were never graded count as 0 and are reported as judged@5 (a later phase
              that retrieves new items needs a short top-up grading round).
  folder name category of single-category photos (as in Phase 1), labels or not.

Metrics (95% bootstrap CIs by photo, or by item for graded retrieval)
  count_exact / count_mae      detected vs expected item count
  category_acc_first           single-category photos: first (top-score) item's label == folder
  item_category_f1             per-category count overlap between predicted labels and labelled counts
  graded nDCG@5                absolute: ideal = grade 3 at every rank, so variants are comparable
  P@5 (grade >= 2), success@1 (top-1 grade >= 2), mean grade@5, judged@5
  negatives rejected           all items no_match (or nothing detected), by negative type
  per-tag breakdown of count / category / F1

Usage
  python src/eval/own_eval.py                       # baseline, holdout (the reporting half)
  python src/eval/own_eval.py --half tune           # for choosing settings in later phases
  python src/eval/own_eval.py --run                 # re-run production pipeline first (e.g. after adding photos)
  python src/eval/own_eval.py --detections <file> --tag <variant> --category-field fused_category
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import (STUMPER_SPLIT_PATH, RESULTS_DIR, bootstrap_ci, fmt_ci, load_json, save_json,
                             replace_log_section)

OWN = "data/eval_own"
BASELINE_DETS = f"{OWN}/baseline_detections.json"
CATS = config.CATEGORIES
TAGS = ["chain", "overlap", "hand_held", "pendant", "blur", "low_light", "reflection", "multi_item"]


OOS_WORDS = ("ring", "anklet", "watch", "brooch", "nose", "toe", "oos_")   # same rule as prepare_own_eval


def negative_type_guess(photo):
    """Unlabelled negatives: 'oos_' filename prefix (or an obvious word) -> out-of-scope jewellery."""
    name = os.path.basename(photo).lower()
    return "out_of_scope_jewellery" if any(w in name for w in OOS_WORDS) else "non_jewellery"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_labels():
    path = f"{OWN}/labels.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        if r["verified"].strip() not in ("1", "1.0"):
            continue
        n = {c: int(float(r[f"n_{c}"] or 0)) for c in CATS}
        tags = {t for t in r["failure_tags"].split(";") if t}
        if sum(n.values()) >= 2:
            tags.add("multi_item")
        out[r["photo"]] = {"n": n, "total": sum(n.values()), "tags": tags,
                           "negative_type": r["negative_type"], "folder": r["folder"]}
    return out


def load_grades():
    """grades.csv plus grades_topup_*.csv / grades_pool_*.csv; extra grades merge into the matching item
    (same photo, IoU >= 0.9) or add a new item (e.g. tune items from a pooled page)."""
    import glob
    paths = ([f"{OWN}/grades.csv"] + sorted(glob.glob(f"{OWN}/grades_topup_*.csv"))
             + sorted(glob.glob(f"{OWN}/grades_pool_*.csv")))
    items = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for it in _read_grade_file(path):
            same = next((x for x in items if x["photo"] == it["photo"] and iou(x["box"], it["box"]) >= 0.9), None)
            if same is None:
                items.append(it)
            else:
                # first grade wins: a later page may re-show a result (e.g. pre-selected on a top-up page);
                # the original judgement is kept and later files only add results not graded before
                for c, g in it["grades"].items():
                    same["grades"].setdefault(c, g)
                same["bad_crop"] = same["bad_crop"] or it["bad_crop"]
    return items


def _read_grade_file(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    items = []
    for _, r in df.iterrows():
        ks = [int(c.split("_")[-1]) for c in df.columns if c.startswith("catalogue_id_")]
        g = {r[f"catalogue_id_{k}"]: int(r[f"grade_{k}"]) for k in ks
             if r[f"catalogue_id_{k}"] and r[f"grade_{k}"] != ""}
        if not g and r["bad_crop"] not in ("1", "1.0"):
            continue                                   # not graded yet
        items.append({"photo": r["photo"], "box": [float(r[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")],
                      "bad_crop": r["bad_crop"] in ("1", "1.0"), "grades": g})
    return items


def run_production(split, out_path):
    from src.eval.prepare_own_eval import run_detections
    dets = run_detections(split)
    save_json(dets, out_path)
    return dets


def first_item(items):
    return max(items, key=lambda it: it["detector_score"]) if items else None


def f1_counts(pred, exp):
    tp = sum(min(pred.get(c, 0), exp.get(c, 0)) for c in CATS)
    npred, nexp = sum(pred.values()), sum(exp.values())
    p = tp / npred if npred else (1.0 if nexp == 0 else 0.0)
    r = tp / nexp if nexp else 1.0
    return (2 * p * r / (p + r) if p + r else 0.0), tp, npred, nexp


def evaluate(dets, split, labels, graded, half, cat_field):
    photos = [p for p, v in split.items() if half == "all" or v["half"] == half]
    res = {"half": half, "category_field": cat_field, "n_photos": len(photos)}

    # ---- per-photo records
    recs = []
    for p in photos:
        d = dets.get(p)
        if d is None:
            continue
        folder = split[p]["folder"]
        items = d["items"]
        lab = labels.get(p)
        rec = {"photo": p, "folder": folder, "n_det": len(items), "labelled": lab is not None,
               "tags": sorted(lab["tags"]) if lab else [],
               "rejected": all(it["status"] == "no_match" for it in items)}
        fi = first_item(items)
        if folder in CATS:
            rec["first_ok"] = float(fi is not None and fi.get(cat_field) == folder) if fi else 0.0
            rec["first_detected"] = fi is not None
        if lab and folder != "negatives":
            pred = Counter(it.get(cat_field) for it in items)
            rec["count_exact"] = float(len(items) == lab["total"])
            rec["count_err"] = len(items) - lab["total"]
            rec["f1"], *_ = f1_counts(pred, lab["n"])
        if folder == "negatives":
            rec["negative_type"] = (lab or {}).get("negative_type") or negative_type_guess(p)
        recs.append(rec)
    missing = [p for p in photos if p not in dets]
    if missing:
        res["missing_photos"] = missing

    pos = [r for r in recs if r["folder"] != "negatives"]
    single = [r for r in recs if r["folder"] in CATS and r.get("first_detected")]
    lab_pos = [r for r in pos if "count_exact" in r]
    neg = [r for r in recs if r["folder"] == "negatives"]

    res["detection_recall"] = bootstrap_ci([float(r["n_det"] > 0) for r in pos])
    res["category_acc_first"] = bootstrap_ci([r["first_ok"] for r in single])
    res["category_acc_first_by_folder"] = {c: bootstrap_ci([r["first_ok"] for r in single if r["folder"] == c]) for c in CATS}
    res["n_labelled_positive"] = len(lab_pos)
    res["count_exact"] = bootstrap_ci([r["count_exact"] for r in lab_pos])
    res["count_mae"] = bootstrap_ci([abs(r["count_err"]) for r in lab_pos])
    res["over_count_rate"] = float(np.mean([r["count_err"] > 0 for r in lab_pos])) if lab_pos else None
    res["under_count_rate"] = float(np.mean([r["count_err"] < 0 for r in lab_pos])) if lab_pos else None
    res["item_category_f1"] = bootstrap_ci([r["f1"] for r in lab_pos])
    res["negatives_rejected"] = bootstrap_ci([float(r["rejected"]) for r in neg])
    res["negatives_rejected_by_type"] = {t: bootstrap_ci([float(r["rejected"]) for r in neg if r["negative_type"] == t])
                                         for t in sorted({r["negative_type"] for r in neg})}

    by_tag = {}
    for t in TAGS + ["no_tag"]:
        sel = [r for r in lab_pos if (t in r["tags"]) or (t == "no_tag" and not r["tags"])]
        sel_single = [r for r in sel if r["folder"] in CATS and r.get("first_detected")]
        if sel:
            by_tag[t] = {"n": len(sel), "count_exact": bootstrap_ci([r["count_exact"] for r in sel]),
                         "item_category_f1": bootstrap_ci([r["f1"] for r in sel]),
                         "category_acc_first": bootstrap_ci([r["first_ok"] for r in sel_single])}
    res["by_tag"] = by_tag

    # ---- graded retrieval (holdout items only)
    g_items = [g for g in graded if half == "all" or split.get(g["photo"], {}).get("half") == half]
    rows, n_bad, n_unmatched = [], 0, 0
    disc = 1.0 / np.log2(np.arange(2, 7))
    for g in g_items:
        if g["bad_crop"]:
            n_bad += 1
            continue
        items = dets.get(g["photo"], {}).get("items", [])
        best = max(items, key=lambda it: iou(it["box"], g["box"]), default=None)
        if best is None or iou(best["box"], g["box"]) < 0.5:
            n_unmatched += 1
            rows.append({"ndcg5": 0.0, "p5": 0.0, "s1": 0.0, "mean_grade": 0.0, "judged": 1.0})
            continue
        ids = [r["catalogue_id"] for r in best["top5"][:5]]
        gains = [g["grades"].get(c, 0) for c in ids] + [0] * (5 - len(ids))
        judged = [c in g["grades"] for c in ids]
        rows.append({"ndcg5": float(np.dot(gains, disc) / (3 * disc.sum())),
                     "p5": float(np.mean([x >= 2 for x in gains])), "s1": float(gains[0] >= 2),
                     "mean_grade": float(np.mean(gains)), "judged": float(np.mean(judged)) if judged else 0.0})
    res["graded"] = {"n_items": len(rows), "n_bad_crop": n_bad, "n_unmatched": n_unmatched,
                     **{k: bootstrap_ci([r[k] for r in rows]) for k in ("ndcg5", "p5", "s1", "mean_grade", "judged")}}
    return res


def markdown(res, tag, dets_path):
    g = res["graded"]
    lines = [f"## Phase 3 — own-photo eval, {tag}, half = {res['half']} ({datetime.now().isoformat(timespec='seconds')})", "",
             f"Detections: `{dets_path}`; category label: `{res['category_field']}`. 95% bootstrap CIs. "
             f"Labelled (verified) positive photos in this half: {res['n_labelled_positive']}. "
             f"Graded nDCG@5 is absolute (ideal = grade 3 at every rank).", "",
             "| metric | value |", "|---|---|",
             f"| Detection recall | {fmt_ci(res['detection_recall'])} |",
             f"| Category acc (first item vs folder) | {fmt_ci(res['category_acc_first'])} |",
             f"| Item-category F1 (vs labelled counts) | {fmt_ci(res['item_category_f1'])} |",
             f"| Count exact | {fmt_ci(res['count_exact'])} (over {res['over_count_rate'] or 0:.2f} / under {res['under_count_rate'] or 0:.2f}) |",
             f"| Count MAE | {fmt_ci(res['count_mae'], 2)} |",
             f"| Graded nDCG@5 (n={g['n_items']}, bad crops {g['n_bad_crop']}, unmatched {g['n_unmatched']}) | {fmt_ci(g['ndcg5'])} |",
             f"| P@5 (grade ≥ 2) | {fmt_ci(g['p5'])} |",
             f"| Success@1 (top-1 grade ≥ 2) | {fmt_ci(g['s1'])} |",
             f"| Mean grade@5 | {fmt_ci(g['mean_grade'], 2)} |",
             f"| judged@5 | {fmt_ci(g['judged'])} |",
             f"| Negatives rejected | {fmt_ci(res['negatives_rejected'])} |"]
    for t, v in res["negatives_rejected_by_type"].items():
        lines.append(f"| … {t} | {fmt_ci(v)} |")
    lines += ["", "| tag | n | count exact | item-category F1 | category acc (first) |", "|---|---|---|---|---|"]
    for t, v in res["by_tag"].items():
        lines.append(f"| {t} | {v['n']} | {fmt_ci(v['count_exact'])} | {fmt_ci(v['item_category_f1'])} | {fmt_ci(v['category_acc_first'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["holdout", "tune", "all"], default="holdout")
    ap.add_argument("--detections", default=BASELINE_DETS)
    ap.add_argument("--category-field", default="classifier_category",
                    help="classifier_category (production), detector_category (what the UI shows), or a later fused field")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--run", action="store_true", help="re-run the production pipeline into --detections first")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    if args.run:
        dets = run_production(split, args.detections)
    else:
        dets = load_json(args.detections)
    labels, graded = load_labels(), load_grades()
    print(f"labels verified: {len(labels)} photos | graded items: {len(graded)}")
    if not labels:
        print("[note] no verified labels yet — count / F1 / per-tag metrics will be empty")

    res = evaluate(dets, split, labels, graded, args.half, args.category_field)
    if res.get("missing_photos"):
        print(f"[warn] {len(res['missing_photos'])} photos in the split have no detections — rerun with --run")
    out = {"created": datetime.now().isoformat(timespec="seconds"), "detections": args.detections, **res}
    path = os.path.join(RESULTS_DIR, f"own_eval_{args.tag}_{args.half}_{args.category_field}.json")
    save_json(out, path)
    md = markdown(res, f"{args.tag} [{args.category_field}]", args.detections)
    if not args.no_log:
        replace_log_section(f"## Phase 3 — own-photo eval, {args.tag} [{args.category_field}], half = {args.half} ", md)
    print(md)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
