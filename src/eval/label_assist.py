"""
Phase 8a — label drafting assist, folder + pick-a-number version.
Images organized in subfolders by category (earring/necklace/bracelet),
plus mixed/ (multi-item) and negatives/ (not in catalogue). Folder name
auto-fills category and is_negative — you only pick a number per item.
"""

import os
import csv
import base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
from io import BytesIO
import sys
sys.path.append(".")
from src.pipeline.run_pipeline import run_pipeline, load_all_models

STUMPER_ROOT = "data/stumper/images"
FOLDERS = ["earring", "necklace", "bracelet", "mixed", "negatives"]
DRAFT_CSV = "data/stumper/labels_draft.csv"
REVIEW_HTML = "data/stumper/review_report.html"


def image_to_base64(image, max_size=160):
    image = image.copy()
    image.thumbnail((max_size, max_size))
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    load_all_models()
    os.makedirs(os.path.dirname(DRAFT_CSV), exist_ok=True)

    rows = []
    html = ["""<html><head><style>
    body{font-family:sans-serif;margin:20px;}
    h2{background:#333;color:white;padding:8px 12px;margin-top:30px;position:sticky;top:0;}
    .row{display:flex;gap:10px;align-items:flex-start;border-bottom:2px solid #eee;padding:12px 0;flex-wrap:wrap;}
    .thumb{text-align:center;} .thumb img{width:130px;height:130px;object-fit:cover;border:2px solid #ccc;}
    .num{font-weight:bold;font-size:18px;color:#333;} .caption{font-size:10px;white-space:pre-line;max-width:130px;}
    .query img{border:3px solid #333;}
    </style></head><body><h1>Pick the correct match by number in labels_draft.csv</h1>"""]

    total_count = 0

    for folder in FOLDERS:
        folder_path = os.path.join(STUMPER_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"[skip] {folder_path} doesn't exist")
            continue

        images = sorted(f for f in os.listdir(folder_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not images:
            continue

        print(f"{folder}: {len(images)} images")
        html.append(f'<h2>{folder.upper()} ({len(images)} photos)</h2>')
        is_negative_folder = (folder == "negatives")

        for fname in images:
            total_count += 1
            path = os.path.join(folder_path, fname)
            image = Image.open(path).convert("RGB")
            items = run_pipeline(image, top_k=5)

            html.append(f'<h4>{fname} — {len(items)} item(s)</h4>')

            if not items:
                rows.append({
                    "folder": folder, "photo_filename": fname, "item_index": "",
                    "chosen_rank": "", "manual_catalogue_id": "",
                    "failure_conditions": "", "is_negative": str(is_negative_folder).lower(),
                })
                html.append('<p>No detections.</p>')
                continue

            for item in items:
                html.append('<div class="row">')
                html.append(f'<div class="thumb query"><img src="data:image/jpeg;base64,'
                            f'{image_to_base64(item["crop"])}"><div class="caption">QUERY\nitem {item["item_index"]}</div></div>')

                row = {
                    "folder": folder, "photo_filename": fname, "item_index": item["item_index"],
                    "chosen_rank": "", "manual_catalogue_id": "",
                    "failure_conditions": "", "is_negative": str(is_negative_folder).lower(),
                }

                for rank, r in enumerate(item["results"][:5], 1):
                    try:
                        result_img = Image.open(r["image_path"]).convert("RGB")
                        b64 = image_to_base64(result_img)
                        shop = r["catalogue_id"].split("_")[1] if "_" in r["catalogue_id"] else "?"
                        html.append(f'<div class="thumb"><div class="num">#{rank}</div>'
                                    f'<img src="data:image/jpeg;base64,{b64}">'
                                    f'<div class="caption">{r["category"]}\nsim={r["score"]:.3f}\n{shop}</div></div>')
                    except Exception:
                        pass
                    row[f"candidate_{rank}_id"] = r["catalogue_id"]

                rows.append(row)
                html.append('</div>')

    html.append("</body></html>")
    with open(REVIEW_HTML, "w") as f:
        f.write("\n".join(html))

    fieldnames = ["folder", "photo_filename", "item_index", "chosen_rank", "manual_catalogue_id",
                  "failure_conditions", "is_negative"] + [f"candidate_{i}_id" for i in range(1, 6)]
    with open(DRAFT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal photos: {total_count}")
    print(f"Draft CSV: {DRAFT_CSV}")
    print(f"Review page: {REVIEW_HTML} — organized by folder, scroll through each section")
    print(f"\nFor each row, fill in:")
    print(f"  chosen_rank = 1-5  (correct match's number)")
    print(f"  chosen_rank = 0    (none of the 5 correct — write real ID in manual_catalogue_id)")
    print(f"  negatives/ rows already have is_negative=true pre-filled — nothing to pick")


if __name__ == "__main__":
    main()