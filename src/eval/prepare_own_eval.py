"""
Phase 3 — prepare the own-photo evaluation set so the manual part is quick.

Runs the CURRENT production pipeline (preprocess_image -> run_pipeline, top 20
-> first 5 shown, as the API does) once on every stumper photo, caches the
detections, and generates:

  data/eval_own/labels.csv   prefilled per-photo labels (verified=0 until you check them)
  data/eval_own/label.html   click-through labelling page: photo with detected boxes,
                             per-category counts, failure tags, notes -> "Download labels.csv"
  data/eval_own/grades.csv   ~50 holdout photo-items x current top-5 (grades blank)
  data/eval_own/grade.html   grading page: query crop beside its top-5, 0-3 buttons
                             -> "Download grades.csv"
  data/eval_own/baseline_detections.json   cached detections (boxes, labels, top-5)
  data/eval_own/grade_items.json           the exact items/boxes that were graded

Photo sets
  labels: 60 "core" positive photos (all 13 mixed + folder-stratified, ~30 tune / ~30
          holdout), then the remaining positives as "optional", then every negative.
  grades: holdout photos only (own_eval reports on holdout): up to 2 items per mixed
          photo + a folder-stratified sample of the top item of single-category photos.

Usage: python src/eval/prepare_own_eval.py [--force]   (refuses to overwrite labels/grades CSVs)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.append(".")
import config
from src.eval.common import STUMPER_SPLIT_PATH, Progress, load_json, save_json, thumb_b64
from src.pipeline.run_pipeline import run_pipeline, load_all_models
from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.detect import crop_with_padding
from src.api.schemas import extract_shop_name

OUT = "data/eval_own"
SEED = 42
TAGS = ["chain", "overlap", "hand_held", "pendant", "blur", "low_light", "reflection"]
CORE_TARGET = {"holdout": {"earring": 13, "necklace": 5, "bracelet": 5},
               "tune": {"earring": 13, "necklace": 5, "bracelet": 6}}       # + every mixed photo
GRADE_TARGET = {"earring": 25, "necklace": 8, "bracelet": 9}                # + up to 2 items per mixed photo
OOS_WORDS = ("ring", "anklet", "watch", "brooch", "nose", "toe", "oos_")
BOX_COLORS = {"earring": (214, 51, 132), "necklace": (52, 120, 246), "bracelet": (40, 167, 69)}


def run_detections(split):
    load_all_models()
    out = {}
    prog = Progress(len(split), "pipeline")
    for i, (photo, info) in enumerate(split.items()):
        img = load_and_preprocess(photo)
        items = run_pipeline(img, top_k=20)
        out[photo] = {"folder": info["folder"], "half": info["half"], "size": list(img.size), "items": [
            {"item_index": it["item_index"], "box": [float(v) for v in it["box"]],
             "detector_category": it["detector_category"], "detector_score": float(it["detector_score"]),
             "classifier_category": it["classifier_category"], "status": it["decision"]["status"],
             "similarity": float(it["decision"]["similarity"]),
             "top5": [{"catalogue_id": r["catalogue_id"], "category": r["category"],
                       "image_path": r["image_path"], "score": float(r["score"])} for r in it["results"][:5]]}
            for it in items]}
        prog.update(i + 1)
    return out


def boxed_b64(photo, items, max_side=520, highlight=None):
    img = load_and_preprocess(photo)
    scale = max_side / max(img.size)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    d = ImageDraw.Draw(img)
    for it in items:
        if highlight is not None and it["item_index"] != highlight:
            continue
        x0, y0, x1, y1 = [v * scale for v in it["box"]]
        col = BOX_COLORS.get(it["detector_category"], (255, 160, 0))
        d.rectangle((x0, y0, x1, y1), outline=col, width=3)
        d.rectangle((x0, y0, x0 + 18, y0 + 14), fill=col)
        d.text((x0 + 4, y0 + 1), str(it["item_index"] + 1), fill=(255, 255, 255))
    return thumb_b64(img, max_side)


def pick_core(dets):
    rng = np.random.default_rng(SEED)
    core = [p for p, d in dets.items() if d["folder"] == "mixed"]
    for half, targets in CORE_TARGET.items():
        for folder, k in targets.items():
            pool = sorted(p for p, d in dets.items() if d["folder"] == folder and d["half"] == half)
            core += [pool[i] for i in sorted(rng.choice(len(pool), size=min(k, len(pool)), replace=False))]
    return core


def pick_grade_items(dets, half="holdout", targets=None):
    targets = targets or GRADE_TARGET
    rng = np.random.default_rng(SEED)
    items = []
    for p, d in sorted(dets.items()):
        if d["half"] == half and d["folder"] == "mixed":
            items += [(p, it) for it in sorted(d["items"], key=lambda x: -x["detector_score"])[:2]]
    for folder, k in targets.items():
        pool = sorted(p for p, d in dets.items() if d["half"] == half and d["folder"] == folder and d["items"])
        for i in sorted(rng.choice(len(pool), size=min(k, len(pool)), replace=False)):
            items.append((pool[i], max(dets[pool[i]]["items"], key=lambda x: x["detector_score"])))
    return items


def negative_type_guess(photo):
    name = os.path.basename(photo).lower()
    return "out_of_scope_jewellery" if any(w in name for w in OOS_WORDS) else "non_jewellery"


# ------------------------------------------------------------------ labels

LABEL_COLS = ["photo", "folder", "half", "priority", "detected_count", "n_earring", "n_necklace", "n_bracelet",
              "expected_item_count", "failure_tags", "negative_type", "notes", "verified"]


def label_rows(dets, core):
    rows = []
    order = core + sorted(p for p, d in dets.items() if p not in core and d["folder"] != "negatives") \
        + sorted(p for p, d in dets.items() if d["folder"] == "negatives")
    for p in order:
        d = dets[p]
        n = {c: 0 for c in config.CATEGORIES}
        if d["folder"] in config.CATEGORIES:
            n[d["folder"]] = len(d["items"])                       # prefill: detector count, folder category
        elif d["folder"] == "mixed":
            n.update(Counter(it["detector_category"] for it in d["items"]))
        neg = d["folder"] == "negatives"
        rows.append({"photo": p, "folder": d["folder"], "half": d["half"],
                     "priority": "negative" if neg else ("core" if p in core else "optional"),
                     "detected_count": len(d["items"]), "n_earring": n["earring"], "n_necklace": n["necklace"],
                     "n_bracelet": n["bracelet"], "expected_item_count": 0 if neg else sum(n.values()),
                     "failure_tags": "", "negative_type": negative_type_guess(p) if neg else "",
                     "notes": "", "verified": 0})
    return rows


def label_html(dets, rows):
    cards = []
    for i, r in enumerate(rows):
        d = dets[r["photo"]]
        det_txt = ", ".join(f"{c}×{k}" for c, k in Counter(it["detector_category"] for it in d["items"]).items()) or "nothing"
        if r["priority"] == "negative":
            body = (f'<label>Type <select data-f="negative_type"><option value="non_jewellery">non-jewellery object</option>'
                    f'<option value="out_of_scope_jewellery">out-of-scope jewellery (ring, anklet, watch, nose pin…)</option></select></label>')
        else:
            nums = "".join(f'<label>{c}s <input type="number" min="0" max="20" data-f="n_{c}"></label>' for c in config.CATEGORIES)
            tags = "".join(f'<label class="tag"><input type="checkbox" data-tag="{t}">{t}</label>' for t in TAGS)
            body = f'<div class="nums">{nums}<span class="total"></span></div><div class="tags">{tags}</div>'
        cards.append(
            f'<div class="card {r["priority"]}" data-i="{i}"><img src="{boxed_b64(r["photo"], d["items"])}">'
            f'<div class="info"><div class="hd"><b>{os.path.basename(r["photo"])}</b> · {r["folder"]} · {r["half"]} · '
            f'<span class="pri">{r["priority"]}</span></div><div class="det">Detector found {len(d["items"])}: {det_txt}</div>'
            f'{body}<input class="notes" data-f="notes" placeholder="notes (optional)">'
            f'<button class="ver">Mark verified</button></div></div>')
    return PAGE.format(title="Own-photo labels", key="jl_phase3_labels", csv_name="labels.csv",
                       intro=LABEL_INTRO, cards="".join(cards), data=json.dumps(rows),
                       cols=json.dumps(LABEL_COLS), script=LABEL_JS)


LABEL_INTRO = """<b>Own-photo labels</b> — for each photo: count the <i>separate pieces really visible</i>
(a pair of earrings = 2, one chain = 1), tick any failure tags, then <b>Mark verified</b>. Counts are prefilled from
the detector (boxes drawn on the photo), so they are often wrong on purpose-built hard photos — check them.
<b>Core</b> photos (first 60) are needed; <b>optional</b> ones improve the estimates; negatives only need their type checked.
<br>Tags: <i>chain</i> long chain/necklace · <i>overlap</i> pieces overlap/touch · <i>hand_held</i> held in hand ·
<i>pendant</i> pendant necklace · <i>blur</i> · <i>low_light</i> · <i>reflection</i> glare/shiny reflections.
(<i>multi_item</i> is set automatically when the total is ≥ 2.)"""

LABEL_JS = """
function load(card,row){card.querySelectorAll('[data-f]').forEach(el=>{el.value=row[el.dataset.f]??''});
 const tags=(row.failure_tags||'').split(';').filter(Boolean);card.querySelectorAll('[data-tag]').forEach(cb=>cb.checked=tags.includes(cb.dataset.tag));
 card.classList.toggle('done',+row.verified===1);total(card,row)}
function total(card,row){const t=card.querySelector('.total');if(!t)return;const n=['n_earring','n_necklace','n_bracelet'].reduce((s,k)=>s+(+row[k]||0),0);
 row.expected_item_count=n;t.textContent='total '+n}
function bind(card,row){card.querySelectorAll('[data-f]').forEach(el=>el.oninput=()=>{row[el.dataset.f]=el.value;total(card,row);save()});
 card.querySelectorAll('[data-tag]').forEach(cb=>cb.onchange=()=>{row.failure_tags=[...card.querySelectorAll('[data-tag]:checked')].map(c=>c.dataset.tag).join(';');save()});
 card.querySelector('.ver').onclick=()=>{row.verified=+row.verified?0:1;card.classList.toggle('done',+row.verified===1);save()}}
function progress(){const by={};rows.forEach(r=>{by[r.priority]=by[r.priority]||[0,0];by[r.priority][1]++;by[r.priority][0]+=+r.verified===1});
 return Object.entries(by).map(([k,v])=>`${k}: ${v[0]}/${v[1]} verified`).join(' · ')}
"""

# ------------------------------------------------------------------ grades

def grade_cols(k):
    return (["photo", "half", "item_index", "box_x0", "box_y0", "box_x1", "box_y1", "detector_category", "bad_crop"]
            + [f"catalogue_id_{i}" for i in range(1, k + 1)] + [f"grade_{i}" for i in range(1, k + 1)])


def grade_rows(items, meta, half="holdout", k_max=5):
    rows = []
    for p, it in items:
        r = {"photo": p, "half": half, "item_index": it["item_index"],
             "box_x0": round(it["box"][0], 1), "box_y0": round(it["box"][1], 1),
             "box_x1": round(it["box"][2], 1), "box_y1": round(it["box"][3], 1),
             "detector_category": it["detector_category"], "bad_crop": 0}
        for k in range(k_max):
            res = it["top5"][k] if k < len(it["top5"]) else None
            r[f"catalogue_id_{k + 1}"] = res["catalogue_id"] if res else ""
            r[f"grade_{k + 1}"] = ""
        rows.append(r)
    return rows


GRADE_COLS = grade_cols(5)


def grade_html(items, rows, meta, key="jl_phase3_grades", csv_name="grades.csv", intro=None, blind=False):
    """blind=True (pooled pages): no rank numbers or scores, so the grader cannot tell which system found a result."""
    title = dict(zip(meta["catalogue_id"], meta["title"]))
    url = dict(zip(meta["catalogue_id"], meta["listing_url"]))
    cards = []
    for i, ((p, it), r) in enumerate(zip(items, rows)):
        img = load_and_preprocess(p)
        crop = crop_with_padding(img, it["box"])
        res_cells = []
        for k, res in enumerate(it["top5"]):
            cid = res["catalogue_id"]
            btns = "".join(f'<button data-g="{g}">{g}</button>' for g in range(4))
            meta_line = (f"{extract_shop_name(cid)} · {res['category']}" if blind
                         else f"{extract_shop_name(cid)} · {res['category']} · {res['score']:.2f}")
            res_cells.append(f'<div class="res" data-k="{k + 1}"><a href="{url.get(cid, "#")}" target="_blank">'
                             f'<img src="{thumb_b64(res["image_path"], 150)}"></a><div class="rt">{"" if blind else f"{k + 1}. "}'
                             f'{title.get(cid, cid)[:60]}<br><small>{meta_line}</small></div>'
                             f'<div class="g">{btns}</div></div>')
        cards.append(
            f'<div class="card" data-i="{i}"><div class="q"><img class="crop" src="{thumb_b64(crop, 220)}">'
            f'<img class="ctx" src="{boxed_b64(p, [it], 200, highlight=it["item_index"])}">'
            f'<div class="hd"><b>#{i + 1}</b> {os.path.basename(p)} · item {it["item_index"] + 1}<br>'
            f'<small>detector: {it["detector_category"]}</small></div>'
            f'<label class="bad"><input type="checkbox" data-f="bad_crop"> crop is not a jewellery piece</label></div>'
            f'<div class="results">{"".join(res_cells)}</div></div>')
    return PAGE.format(title="Own-photo grading", key=key, csv_name=csv_name,
                       intro=intro or GRADE_INTRO, cards="".join(cards), data=json.dumps(rows),
                       cols=json.dumps(grade_cols(sum(1 for c in rows[0] if c.startswith("catalogue_id_")))), script=GRADE_JS)


GRADE_INTRO = """<b>Grade each result against the query crop on the left</b> —
<b>3</b> near-identical design · <b>2</b> same design family · <b>1</b> vaguely similar, right category ·
<b>0</b> irrelevant or wrong category. If the crop is not a jewellery piece (skin, background), tick the box instead and skip
its results. Click a result image to open the listing."""

GRADE_JS = """
function load(card,row){card.querySelectorAll('.res').forEach(res=>{const v=row['grade_'+res.dataset.k];
 res.querySelectorAll('button').forEach(b=>b.classList.toggle('on',String(b.dataset.g)===String(v)))});
 const cb=card.querySelector('[data-f="bad_crop"]');cb.checked=+row.bad_crop===1;card.classList.toggle('badc',cb.checked);done(card,row)}
function isDone(row){return +row.bad_crop===1||Object.keys(row).filter(c=>c.startsWith('catalogue_id_')).every(c=>{
 const k=c.split('_').pop();return row['grade_'+k]!==''||row[c]===''})}
function done(card,row){card.classList.toggle('done',isDone(row))}
function bind(card,row){card.querySelectorAll('.res').forEach(res=>res.querySelectorAll('button').forEach(b=>b.onclick=()=>{
 row['grade_'+res.dataset.k]=b.dataset.g;load(card,row);save()}));
 const cb=card.querySelector('[data-f="bad_crop"]');cb.onchange=()=>{row.bad_crop=cb.checked?1:0;load(card,row);save()}}
function progress(){return rows.filter(isDone).length+'/'+rows.length+' items graded'}
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--bg:#fdfaf6;--card:#fff;--ink:#2b2633;--muted:#8a8494;--ok:#e3f5e4;--acc:#b7a3d9;--on:#7b5ea7}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:16px;font-size:13px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid #e7e0ee;z-index:2}}
.intro{{line-height:1.5;margin-bottom:6px}} .bar .p{{margin:0 0 6px;font-weight:600}} button{{font:inherit;cursor:pointer;border:1px solid #ccc;background:#fff;border-radius:6px;padding:3px 10px}}
.card{{background:var(--card);border-radius:12px;padding:10px;margin:12px 0;display:flex;flex-wrap:wrap;gap:12px;box-shadow:0 1px 3px #0001}}
.card.done{{background:var(--ok)}} .card img{{max-width:100%;border-radius:8px}} .card>img{{width:min(520px,100%)}}
.info{{flex:1 1 260px;min-width:0}} .hd{{margin-bottom:6px}} .pri{{color:var(--on)}} .det{{color:var(--muted);margin-bottom:8px}}
.nums label{{display:inline-block;margin:0 10px 6px 0}} .nums input{{width:52px;font:inherit;padding:2px}}
.total{{font-weight:600;margin-left:6px}} .tags label{{display:inline-block;margin:0 10px 4px 0}}
.notes{{width:100%;margin:8px 0;font:inherit;padding:4px;box-sizing:border-box}} .ver{{background:var(--acc);color:#fff;border:none}}
.q{{width:230px;display:flex;flex-direction:column;gap:6px}} .crop{{width:220px;height:220px;object-fit:contain;background:#f4f1ee;border:2px solid var(--on)}}
.ctx{{width:200px}} .results{{flex:1 1 400px;display:flex;flex-wrap:wrap;gap:8px;min-width:0}}
.res{{width:150px}} .res img{{width:150px;height:150px;object-fit:cover}} .rt{{font-size:11px;min-height:44px}}
.g button{{padding:3px 9px;margin-right:2px}} .g button.on{{background:var(--on);color:#fff;border-color:var(--on)}}
.card.badc .results{{opacity:.3}} .bad{{color:#b23}}
</style></head><body>
<div class="intro">{intro}</div><div class="bar"><div class="p" id="prog"></div>
<button id="dl">Download {csv_name}</button> <button id="reset">Reset</button></div>
<div id="cards">{cards}</div>
<script>
const KEY='{key}',COLS={cols};let rows={data};
try{{const s=JSON.parse(localStorage.getItem(KEY)||'null');if(s&&s.length===rows.length)rows=s}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(rows))}}catch(e){{}};document.getElementById('prog').textContent=progress()}}
{script}
document.querySelectorAll('.card').forEach(card=>{{const row=rows[+card.dataset.i];load(card,row);bind(card,row)}});
save();
function esc(v){{v=String(v??'');return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}}
document.getElementById('dl').onclick=()=>{{const csv=[COLS.join(',')].concat(rows.map(r=>COLS.map(c=>esc(r[c])).join(','))).join('\\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='{csv_name}';a.click()}};
document.getElementById('reset').onclick=()=>{{if(confirm('Clear everything on this page?')){{try{{localStorage.removeItem(KEY)}}catch(e){{}};location.reload()}}}};
</script></body></html>"""


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def build_topup(dets_path, name):
    """Top-up grading for a variant: items whose top-5 holds results never graded.
    Existing grades are prefilled, so only the new results need clicks."""
    from src.eval.own_eval import load_grades, iou
    graded = load_grades()
    dets = load_json(dets_path)
    items, rows = [], []
    for g in graded:
        if g["bad_crop"]:
            continue
        cand = dets.get(g["photo"], {}).get("items", [])
        best = max(cand, key=lambda it: iou(it["box"], g["box"]), default=None)
        if best is None or iou(best["box"], g["box"]) < 0.5:
            continue
        if all(r["catalogue_id"] in g["grades"] for r in best["top5"][:5]):
            continue
        items.append((g["photo"], best))
    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    primary = dict(zip(meta["catalogue_id"], meta["primary_image"]))
    for _, it in items:                       # variants evaluated from caches may not carry the matched image
        for r in it["top5"]:
            if not r.get("image_path"):
                r["image_path"] = primary[r["catalogue_id"]]
    rows = grade_rows(items, meta)
    for r, (p, it) in zip(rows, items):
        g = next(x for x in graded if x["photo"] == p and iou(x["box"], it["box"]) >= 0.5)
        for k in range(1, 6):
            if r[f"catalogue_id_{k}"] in g["grades"]:
                r[f"grade_{k}"] = str(g["grades"][r[f"catalogue_id_{k}"]])
    csv_name = f"grades_topup_{name}.csv"
    write_csv(os.path.join(OUT, csv_name), GRADE_COLS, rows)
    intro = (GRADE_INTRO + f"<br><b>Top-up:</b> results you already graded are pre-selected — only grade the "
             f"<b>unselected</b> ones, then download <code>{csv_name}</code>.")
    with open(os.path.join(OUT, f"grade_topup_{name}.html"), "w") as f:
        f.write(grade_html(items, rows, meta, key=f"jl_topup_{name}", csv_name=csv_name, intro=intro))
    n_new = sum(1 for r in rows for k in range(1, 6) if r[f"catalogue_id_{k}"] and r[f"grade_{k}"] == "")
    print(f"top-up: {len(rows)} items, {n_new} new results to grade -> {OUT}/grade_topup_{name}.html")


GRADE_TARGET_TUNE = {"earring": 16, "necklace": 6, "bracelet": 6}                # + up to 2 items per mixed photo


def build_pool(det_paths, half, name):
    """Blind pooled grading: for each item, the union of every system's top-5 (de-duplicated, shuffled,
    no ranks or scores shown), with the listing image that actually matched for the system that found it."""
    from src.eval.own_eval import iou
    systems = [load_json(p) for p in det_paths]
    items = pick_grade_items(systems[0], half=half, targets=GRADE_TARGET_TUNE if half == "tune" else None)
    rng = np.random.default_rng(SEED)
    pooled = []
    for p, it in items:
        seen, res = set(), []
        for sysd in systems:
            other = max(sysd[p]["items"], key=lambda x: iou(x["box"], it["box"]))
            for r in other["top5"][:5]:
                if r["catalogue_id"] not in seen:
                    seen.add(r["catalogue_id"])
                    res.append(r)
        res = [res[i] for i in rng.permutation(len(res))]
        pooled.append((p, {**it, "top5": res}))
    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    k_max = max(len(it["top5"]) for _, it in pooled)
    rows = grade_rows(pooled, meta, half=half, k_max=k_max)
    csv_name = f"grades_pool_{name}.csv"
    write_csv(os.path.join(OUT, csv_name), grade_cols(k_max), rows)
    intro = (GRADE_INTRO + f"<br><b>Pooled, blind page:</b> results from several systems are mixed and shuffled, with no "
             f"ranks or scores — grade each on its own. Then download <code>{csv_name}</code>.")
    with open(os.path.join(OUT, f"grade_pool_{name}.html"), "w") as f:
        f.write(grade_html(pooled, rows, meta, key=f"jl_pool_{name}", csv_name=csv_name, intro=intro, blind=True))
    n = sum(1 for r in rows for k in range(1, k_max + 1) if r[f"catalogue_id_{k}"])
    print(f"pool: {len(rows)} {half} items, {n} results to grade -> {OUT}/grade_pool_{name}.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite labels.csv / grades.csv")
    ap.add_argument("--topup", help="detections file of a variant: build a top-up grading page for its ungraded results")
    ap.add_argument("--name", default="variant", help="name for the top-up files")
    ap.add_argument("--pool", help="comma-separated detections files: build a blind pooled grading page")
    ap.add_argument("--half", default="tune")
    args = ap.parse_args()
    if args.pool:
        return build_pool(args.pool.split(","), args.half, args.name)
    if args.topup:
        return build_topup(args.topup, args.name)
    os.makedirs(OUT, exist_ok=True)
    for f in ("labels.csv", "grades.csv"):
        if os.path.exists(os.path.join(OUT, f)) and not args.force:
            sys.exit(f"{OUT}/{f} exists — it may hold your manual work. Use --force only if you mean to regenerate it.")

    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    dets = run_detections(split)
    save_json(dets, os.path.join(OUT, "baseline_detections.json"))

    core = pick_core(dets)
    rows = label_rows(dets, core)
    write_csv(os.path.join(OUT, "labels.csv"), LABEL_COLS, rows)
    with open(os.path.join(OUT, "label.html"), "w") as f:
        f.write(label_html(dets, rows))

    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    items = pick_grade_items(dets)
    grows = grade_rows(items, meta)
    write_csv(os.path.join(OUT, "grades.csv"), GRADE_COLS, grows)
    save_json([{"photo": p, **it} for p, it in items], os.path.join(OUT, "grade_items.json"))
    with open(os.path.join(OUT, "grade.html"), "w") as f:
        f.write(grade_html(items, grows, meta))

    pri = Counter(r["priority"] for r in rows)
    print(f"labels: {len(rows)} photos ({dict(pri)}) -> {OUT}/labels.csv, label.html")
    print(f"grades: {len(grows)} holdout items × 5 = {sum(1 for r in grows for k in range(1, 6) if r[f'catalogue_id_{k}'])} "
          f"results -> {OUT}/grades.csv, grade.html")
    print("by folder:", dict(Counter(dets[p]['folder'] for p, _ in items)))


if __name__ == "__main__":
    main()
