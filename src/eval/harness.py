"""
Phase 8c — runs the full pipeline over the labelled stumper set
(folder + chosen_rank format), computes Recall@1/5, MRR, Detection
Recall, Category Accuracy (from folder name), no-match precision,
confused-pairs, and per-failure-condition breakdown. Prints a summary
to terminal and writes an HTML report marking each result correct/
incorrect visually.
"""

import os
import base64
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
from io import BytesIO
import pandas as pd
import sys
sys.path.append(".")
from src.pipeline.run_pipeline import run_pipeline, load_all_models

STUMPER_ROOT = "data/stumper/images"
LABELS_CSV = "data/stumper/labels_draft.csv"
EVAL_REPORT_HTML = "data/stumper/eval_report.html"


def image_to_base64(image, max_size=150):
    image = image.copy()
    image.thumbnail((max_size, max_size))
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def resolve_ground_truth(row):
    """chosen_rank 1-5 -> that candidate's id. chosen_rank 0 -> manual_catalogue_id. Blank -> no GT."""
    rank = str(row.get("chosen_rank", "")).strip()
    if rank in ("0",):
        return str(row.get("manual_catalogue_id", "")).strip()
    if rank in ("1", "2", "3", "4", "5"):
        return str(row.get(f"candidate_{rank}_id", "")).strip()
    return ""


def main():
    load_all_models()

    labels = pd.read_csv(LABELS_CSV, dtype=str).fillna("")

    n_positive = n_negative = 0
    n_recall1 = n_recall5 = n_detected = n_correct_no_match = 0
    n_category_correct = n_category_total = 0
    mrr_sum = 0.0
    condition_stats = defaultdict(lambda: {"recall1": 0, "recall5": 0, "total": 0})
    confused_pairs = defaultdict(int)
    unlabeled_skipped = 0

    html = ["""<html><head><style>
    body{font-family:sans-serif;margin:20px;}
    h2{background:#333;color:white;padding:8px 12px;margin-top:30px;position:sticky;top:0;}
    .row{display:flex;gap:10px;align-items:flex-start;border-bottom:2px solid #eee;padding:10px 0;flex-wrap:wrap;}
    .thumb img{width:120px;height:120px;object-fit:cover;border:1px solid #ccc;}
    .caption{font-size:10px;white-space:pre-line;max-width:120px;}
    .ok{color:#2e7d32;font-weight:bold;} .bad{color:#c62828;font-weight:bold;}
    </style></head><body><h1>Eval Report</h1>"""]

    current_folder = None

    for _, row in labels.iterrows():
        folder = row["folder"]
        fname = row["photo_filename"]
        is_negative = str(row["is_negative"]).lower() == "true"
        conditions = [c.strip() for c in row["failure_conditions"].split(";") if c.strip()] or ["none"]

        if folder != current_folder:
            html.append(f'<h2>{folder.upper()}</h2>')
            current_folder = folder

        path = os.path.join(STUMPER_ROOT, folder, fname)
        if not os.path.exists(path):
            print(f"[warn] missing file: {path}")
            continue

        image = Image.open(path).convert("RGB")
        items = run_pipeline(image, top_k=5)

        html.append(f'<div class="row"><b style="width:100%">{fname}</b>')

        if is_negative:
            n_negative += 1
            correctly_rejected = (not items) or all(i["decision"]["status"] == "no_match" for i in items)
            if correctly_rejected:
                n_correct_no_match += 1
            html.append(f'<p class="{"ok" if correctly_rejected else "bad"}">'
                        f'{"Correctly rejected" if correctly_rejected else "FALSE ACCEPT"}</p></div>')
            continue

        gt_id = resolve_ground_truth(row)
        if not gt_id:
            unlabeled_skipped += 1
            html.append('<p style="color:#999">Not labeled yet — skipped</p></div>')
            continue

        n_positive += 1
        if items:
            n_detected += 1

        r1 = any(gt_id in [r["catalogue_id"] for r in i["results"][:1]] for i in items)
        r5 = any(gt_id in [r["catalogue_id"] for r in i["results"][:5]] for i in items)
        m = 0.0
        for i in items:
            for rank, r in enumerate(i["results"], 1):
                if r["catalogue_id"] == gt_id:
                    m = max(m, 1.0 / rank)

        n_recall1 += int(r1)
        n_recall5 += int(r5)
        mrr_sum += m

        # category accuracy: expected category from folder (skip for "mixed")
        if folder in ("earring", "necklace", "bracelet") and items:
            n_category_total += 1
            pred_category = items[0]["classifier_category"]
            if pred_category == folder:
                n_category_correct += 1
            elif pred_category:
                confused_pairs[(folder, pred_category)] += 1

        for cond in conditions:
            condition_stats[cond]["total"] += 1
            condition_stats[cond]["recall1"] += int(r1)
            condition_stats[cond]["recall5"] += int(r5)

        status_class = "ok" if r1 else ("ok" if r5 else "bad")
        html.append(f'<span class="{status_class}">R@1={r1} R@5={r5} MRR={m:.2f}</span>')
        if items:
            for r in items[0]["results"][:5]:
                mark = "✓" if r["catalogue_id"] == gt_id else ""
                try:
                    img = Image.open(r["image_path"]).convert("RGB")
                    b64 = image_to_base64(img)
                    html.append(f'<div class="thumb"><img src="data:image/jpeg;base64,{b64}">'
                                f'<div class="caption">{mark} sim={r["score"]:.3f}</div></div>')
                except Exception:
                    pass
        html.append('</div>')

    html.append("</body></html>")
    with open(EVAL_REPORT_HTML, "w") as f:
        f.write("\n".join(html))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Positives: {n_positive} | Negatives: {n_negative} | Unlabeled (skipped): {unlabeled_skipped}")
    if n_positive:
        print(f"Recall@1: {n_recall1/n_positive:.3f}")
        print(f"Recall@5: {n_recall5/n_positive:.3f}")
        print(f"MRR: {mrr_sum/n_positive:.3f}")
        print(f"Detection Recall: {n_detected/n_positive:.3f}")
    if n_category_total:
        print(f"Category Accuracy: {n_category_correct/n_category_total:.3f}")
    if n_negative:
        print(f"No-match precision (correct rejections): {n_correct_no_match/n_negative:.3f}")

    print(f"\nConfused pairs (true folder -> predicted category):")
    for (true_cat, pred_cat), count in sorted(confused_pairs.items(), key=lambda x: -x[1]):
        print(f"  {true_cat} -> {pred_cat}: {count}")

    print(f"\nPer-failure-condition breakdown:")
    print(f"{'Condition':<20}{'Recall@1':<12}{'Recall@5':<12}{'N':<6}")
    for cond, stats in sorted(condition_stats.items()):
        r1 = stats["recall1"] / stats["total"] if stats["total"] else 0
        r5 = stats["recall5"] / stats["total"] if stats["total"] else 0
        print(f"{cond:<20}{r1:<12.3f}{r5:<12.3f}{stats['total']:<6}")

    print(f"\nHTML report: {EVAL_REPORT_HTML}")


if __name__ == "__main__":
    main()