"""
Phase 2 — fill blank subtype / colour metadata. Fills blanks only; never
overwrites an existing value. Writes a NEW file; the production parquet is
untouched until promotion is approved.

Order of sources (first one that yields a value wins):
  colour : variant -> the Shopify colour option (color_query, 33 raw strings) mapped
                      to the 16-colour palette; raw string kept in color_query_raw
           title   -> colour / metal / stone words in the title
           model   -> best of {CLIP zero-shot (primary image / mean of images), linear probe
                      on CLIP+DINOv2 listing-mean features fit on train-split known labels},
                      chosen by train accuracy (out-of-fold for the probe); filled only above
                      a calibrated confidence threshold, otherwise left blank.
                      Source is written as 'clip' or 'probe'.
  subtype: original-> existing value (scraper's title keywords), kept as is
           title   -> broader, category-aware title keywords
           model   -> same choice as colour, among that category's subtypes

Calibration (no test data, no stumper photos): listings that already have a
non-CLIP label are "known". On TRAIN-split known listings we pick (a) whether to use
the primary image or the mean of all the listing's images and (b) the lowest
confidence threshold whose precision >= --target-precision. Precision/coverage
are then reported on VAL-split known listings. Known rows may be easier than
blank rows, so the spot-check page is the real guard.

Platinum and white gold are kept in the palette for title/variant matches but are
not CLIP candidates: in a photo they are indistinguishable from silver.

Usage:
  python src/catalogue/backfill_attributes.py
  python src/catalogue/backfill_attributes.py --subtype-min-conf 0.7 --color-min-conf 0.8   # manual override
Outputs: data/catalogue_metadata_backfilled.parquet, results/phase2.json,
         results/phase2_spotcheck.html, LOG.md entry.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import load_json, save_json, thumb_b64, replace_log_section, SPLITS_PATH
from src.catalogue.build_clip_embeddings import clip_text_embed, OUT_DIR as CLIP_DIR

OUT_PARQUET = "data/catalogue_metadata_backfilled.parquet"
SPOTCHECK_HTML = "results/phase2_spotcheck.html"
CLIP_LOGIT_SCALE = 100.0
SEED = 42

# ---------------------------------------------------------------- colour words

METALS = [("rose gold", r"rose[\s-]*gold"), ("white gold", r"white[\s-]*gold"),
          ("oxidized", r"oxidi[sz]ed"), ("platinum", r"platinum"),
          ("silver", r"silver|sterling|\b925\b"),
          ("gold", r"gold|vermeil|\b(?:14|18|22|24)\s*(?:k|kt|ct)\b")]
STONES = [("multicolour", r"multi[\s-]*colou?r|rainbow"),
          ("red", r"\bred\b|ruby|maroon"), ("green", r"\bgreen\b|emerald|\bmint\b"),
          ("blue", r"\bblue\b|sapphire|turquoise|\bnavy\b"), ("pink", r"\bpink\b|blush"),
          ("black", r"\bblack\b|onyx"), ("white", r"\bwhite\b"),
          ("purple", r"purple|amethyst|lavender"), ("yellow", r"yellow|citrine"),
          ("orange", r"\borange\b")]


def color_from_text(text):
    """Palette colour from free text, or '' when absent/ambiguous.
    Stone/enamel colours beat metal words (most pieces are 'gold plated' boilerplate);
    two or more stone colours -> multicolour; two different metals -> ambiguous ('')."""
    t = str(text or "").lower()
    metals = []
    for name, pat in METALS:
        if re.search(pat, t):
            metals.append(name)
            t = re.sub(pat, " ", t)          # 'rose gold' must not also count as 'gold'
    stones = {name for name, pat in STONES if re.search(pat, t)}
    if "multicolour" in stones or len(stones) >= 2:
        return "multicolour"
    if len(stones) == 1:
        return stones.pop()
    if len(metals) == 1:
        return metals[0]
    return ""

# ---------------------------------------------------------------- subtype words (category-aware, first match wins)

SUBTYPE_RULES = {
    "earring": [("jhumka", r"jhumk[ai]"), ("chandbali", r"chand\s*bali"), ("huggie", r"huggie"),
                ("ear cuff", r"ear[\s-]*cuff"), ("chandelier", r"chandelier"), ("hoop", r"\bhoops?\b"),
                ("stud", r"\bstuds?\b"), ("drop dangle", r"\bdrops?\b|dangl")],
    "necklace": [("choker", r"choker"), ("layered", r"layer"), ("collar", r"\bcollar\b"),
                 ("pendant", r"pendant|locket"), ("statement", r"statement"),
                 ("beaded", r"\bbead(?:s|ed)?\b"), ("long chain", r"long\s+chain"), ("chain", r"\bchain\b")],
    "bracelet": [("kada", r"\bkadas?\b"), ("bangle", r"bangle"), ("cuff", r"\bcuff\b"),
                 ("tennis", r"tennis"), ("charm", r"\bcharms?\b"), ("beaded", r"\bbead(?:s|ed)?\b"),
                 ("link", r"\blinks?\b"), ("chain", r"\bchain\b")],
}


def subtype_from_title(title, category):
    t = str(title or "").lower()
    for name, pat in SUBTYPE_RULES.get(category, []):
        if re.search(pat, t):
            return name
    return ""

# ---------------------------------------------------------------- CLIP prompts

SUBTYPE_DESC = {
    "earring": {"jhumka": "jhumka earrings, bell-shaped Indian dome earrings",
                "stud": "small stud earrings that sit on the earlobe",
                "hoop": "hoop earrings, ring-shaped circular earrings",
                "drop dangle": "drop earrings that dangle below the earlobe",
                "chandbali": "chandbali earrings, crescent moon shaped Indian earrings",
                "huggie": "huggie earrings, small thick hoops hugging the earlobe",
                "ear cuff": "an ear cuff worn on the outer rim of the ear",
                "chandelier": "chandelier earrings with several cascading tiers"},
    "necklace": {"pendant": "a pendant necklace with a single charm on a thin chain",
                 "chain": "a plain chain necklace without a pendant",
                 "choker": "a choker necklace that sits tightly around the neck",
                 "layered": "a layered necklace with several strands",
                 "statement": "a large bold statement necklace",
                 "collar": "a rigid flat collar necklace",
                 "beaded": "a beaded necklace made of beads or pearls",
                 "long chain": "a very long chain necklace"},
    "bracelet": {"chain": "a thin chain bracelet",
                 "cuff": "an open cuff bracelet",
                 "bangle": "a rigid round bangle",
                 "charm": "a charm bracelet with small hanging charms",
                 "tennis": "a tennis bracelet, a single line of small stones",
                 "beaded": "a beaded bracelet",
                 "link": "a bracelet made of large chain links",
                 "kada": "a kada, a thick heavy Indian bangle"},
}
COLOR_DESC = {"gold": "gold-toned", "rose gold": "rose gold", "silver": "silver-toned",
              "oxidized": "oxidized dark silver", "black": "black", "white": "white",
              "red": "red", "pink": "pink", "blue": "blue", "green": "green", "purple": "purple",
              "yellow": "yellow", "orange": "orange", "multicolour": "multicoloured"}
TEMPLATES = ["a product photo of {}.", "a photo of {}.", "{}, jewellery photo."]


def class_text_matrix(descs):
    """Prompt-ensembled, normalised text embedding per class."""
    rows = []
    for d in descs:
        e = clip_text_embed([t.format(d) for t in TEMPLATES]).mean(axis=0)
        rows.append(e / np.linalg.norm(e))
    return np.stack(rows)


def zero_shot(feats, text_mat):
    logits = CLIP_LOGIT_SCALE * feats @ text_mat.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------- calibration


def pick_threshold(conf, correct, target):
    """Lowest threshold whose precision >= target; else the best-precision one with >=10% coverage."""
    best = None
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        sel = conf >= t
        if sel.sum() < 20:
            break
        prec = correct[sel].mean()
        if prec >= target:
            return float(t), float(prec), float(sel.mean()), True
        if sel.mean() >= 0.10 and (best is None or prec > best[1]):
            best = (float(t), float(prec), float(sel.mean()))
    return (*best, False) if best else (0.99, float("nan"), 0.0, False)


def curve(conf, correct):
    out = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sel = conf >= t
        out.append({"t": t, "coverage": float(sel.mean()) if len(sel) else 0.0,
                    "precision": float(correct[sel].mean()) if sel.sum() else None})
    return out

# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-precision", type=float, default=0.85)
    ap.add_argument("--subtype-min-conf", type=float, default=None, help="manual override of the calibrated threshold")
    ap.add_argument("--color-min-conf", type=float, default=None, help="manual override of the calibrated threshold")
    ap.add_argument("--min-class-precision", type=float, default=0.70,
                    help="classes below this train (OOF) precision at the threshold are never filled")
    ap.add_argument("--n-spotcheck", type=int, default=60)
    args = ap.parse_args()

    meta = pd.read_parquet(config.CATALOGUE_METADATA_PATH)
    split = {k: v["split"] for k, v in load_json(SPLITS_PATH)["listings"].items()}
    df = meta.copy()
    df["split"] = df["catalogue_id"].map(split)
    df["subtype"] = df["subtype"].fillna("").astype(str).str.strip()
    df["color_query_raw"] = df["color_query"].fillna("").astype(str)

    # 1) colour from the Shopify variant option
    variant_map = {s: color_from_text(s) for s in sorted(df["color_query_raw"].unique()) if s}
    df["color_query"] = df["color_query_raw"].map(lambda s: variant_map.get(s, ""))
    df["color_source"] = np.where(df["color_query"] != "", "variant", "")
    df["color_conf"] = np.where(df["color_query"] != "", 1.0, np.nan)

    # 2) subtype: keep originals; flag ones inconsistent with the category (not changed)
    valid_sub = {c: set(v["subtypes"]) for c, v in config.SEARCH_FACETS.items()}
    df["subtype_source"] = np.where(df["subtype"] != "", "original", "")
    df["subtype_conf"] = np.where(df["subtype"] != "", 1.0, np.nan)
    df["subtype_inconsistent"] = [bool(s) and s not in valid_sub[c] for s, c in zip(df["subtype"], df["category"])]

    # 3) title rules for remaining blanks
    t_sub = [subtype_from_title(t, c) for t, c in zip(df["title"], df["category"])]
    t_col = [color_from_text(t) for t in df["title"]]
    for col, src, conf, vals in (("subtype", "subtype_source", "subtype_conf", t_sub),
                                 ("color_query", "color_source", "color_conf", t_col)):
        vals = pd.Series(vals, index=df.index)
        m = (df[col] == "") & (vals != "")
        df.loc[m, col] = vals[m]
        df.loc[m, src] = "title"
        df.loc[m, conf] = 1.0

    # 4) listing features: CLIP primary image, CLIP mean of all images, and CLIP+DINOv2 mean (for the probe)
    clip = np.load(os.path.join(CLIP_DIR, "embeddings.npy"))
    cmeta = pd.read_parquet(os.path.join(CLIP_DIR, "meta.parquet"))
    dino = np.load(config.CATALOGUE_EMBEDDINGS_PATH).astype(np.float32)
    dino /= np.linalg.norm(dino, axis=1, keepdims=True)
    fid_of_path = {p: i for i, p in enumerate(cmeta["image_path"])}
    rows_of_cid = cmeta.groupby("catalogue_id").indices      # same row order as the DINOv2 meta
    feats = {"primary": [], "mean": [], "clip+dino_mean": []}
    for cid, prim in zip(df["catalogue_id"], df["primary_image"]):
        idx = rows_of_cid[cid]
        feats["primary"].append(clip[fid_of_path.get(prim, idx[0])])
        c = clip[idx].mean(axis=0); c /= np.linalg.norm(c)
        d = dino[idx].mean(axis=0); d /= np.linalg.norm(d)
        feats["mean"].append(c)
        feats["clip+dino_mean"].append(np.concatenate([c, d]))
    feats = {k: np.stack(v) for k, v in feats.items()}

    report = {"created": datetime.now().isoformat(timespec="seconds"), "variant_color_map": variant_map,
              "target_precision": args.target_precision, "attributes": {}}
    color_classes = list(COLOR_DESC)

    def groups(attr):
        if attr == "subtype":
            return [(cat, np.where(df["category"] == cat)[0]) for cat in config.CATEGORIES]
        return [("all", np.arange(len(df)))]

    def clip_predict(attr, feat_name):
        """Zero-shot CLIP -> (pred list, conf array, top2 list) for every row."""
        f = feats[feat_name]
        preds, confs, top2 = [""] * len(df), np.zeros(len(df)), [""] * len(df)
        for cat in config.CATEGORIES:
            rows = np.where(df["category"] == cat)[0]
            if attr == "color":
                names = color_classes
                tm = class_text_matrix([f"a {COLOR_DESC[c]} {cat}" for c in names])
            else:
                names = list(SUBTYPE_DESC[cat])
                tm = class_text_matrix([SUBTYPE_DESC[cat][n] for n in names])
            for r, pr in zip(rows, zero_shot(f[rows], tm)):
                o = np.argsort(-pr)
                preds[r], confs[r], top2[r] = names[o[0]], pr[o[0]], f"{names[o[1]]} {pr[o[1]]:.2f}"
        return preds, confs, top2

    def probe_predict(attr, col, train_known):
        """Linear probe on CLIP+DINOv2 listing features, fit on train-split known labels only.
        Train rows get out-of-fold probabilities so the threshold is calibrated honestly."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        f = feats["clip+dino_mean"]
        y = df[col].to_numpy()
        preds, confs, top2 = [""] * len(df), np.zeros(len(df)), [""] * len(df)
        for _, rows in groups(attr):
            tr = rows[train_known[rows]]
            counts = pd.Series(y[tr]).value_counts()
            tr = tr[np.isin(y[tr], counts[counts >= 5].index)]
            clf = LogisticRegression(C=3, max_iter=3000, class_weight="balanced")
            oof = cross_val_predict(clf, f[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                    method="predict_proba")
            clf.fit(f[tr], y[tr])
            p_all = clf.predict_proba(f[rows])
            pos = {r: i for i, r in enumerate(rows)}
            for i, r in enumerate(tr):
                p_all[pos[r]] = oof[i]
            for r, pr in zip(rows, p_all):
                o = np.argsort(-pr)
                preds[r], confs[r], top2[r] = clf.classes_[o[0]], pr[o[0]], f"{clf.classes_[o[1]]} {pr[o[1]]:.2f}"
        return preds, confs, top2

    for attr, col, override in (("subtype", "subtype", args.subtype_min_conf), ("color", "color_query", args.color_min_conf)):
        src_col = f"{attr}_source"
        if attr == "color":
            known = df[src_col].isin(["variant", "title"])
        else:
            known = df[src_col].isin(["original", "title"]) & ~df["subtype_inconsistent"]
        tr, va = (known & (df["split"] == "train")).to_numpy(), (known & (df["split"] == "val")).to_numpy()
        options = {}
        candidates = [(f"zeroshot_{k}", lambda k=k: clip_predict(attr, k)) for k in ("primary", "mean")]
        candidates.append(("probe_clip+dino_mean", lambda: probe_predict(attr, col, tr)))
        for name, fn in candidates:
            preds, confs, top2 = fn()
            correct = np.array([p == yv for p, yv in zip(preds, df[col])])
            options[name] = dict(preds=preds, confs=confs, top2=top2, correct=correct,
                                 train_acc=float(correct[tr].mean()), val_acc=float(correct[va].mean()))
        method = max(options, key=lambda k: options[k]["train_acc"])   # chosen on train (OOF for the probe)
        source = "probe" if method.startswith("probe") else "clip"
        o = options[method]
        t, prec_tr, cov_tr, reached = pick_threshold(o["confs"][tr], o["correct"][tr], args.target_precision)
        if override is not None:
            t = override
        # per-class gate: a class whose train (OOF) precision at t is poor is never used to fill
        preds_arr = np.array(o["preds"], dtype=object)
        class_prec = {}
        for c in sorted(set(preds_arr[tr])):
            s_ = tr & (o["confs"] >= t) & (preds_arr == c)
            class_prec[c] = (float(o["correct"][s_].mean()) if s_.sum() else None, int(s_.sum()))
        blocked = {c for c, (pr, n_) in class_prec.items() if pr is None or n_ < 10 or pr < args.min_class_precision}
        ok = ~np.isin(preds_arr, list(blocked))
        sel_va = (o["confs"][va] >= t) & ok[va]
        fill = ((df[col] == "") & (o["confs"] >= t) & ok).to_numpy()
        df.loc[fill, col] = np.array(o["preds"], dtype=object)[fill]
        df.loc[fill, src_col] = source
        df.loc[fill, f"{attr}_conf"] = o["confs"][fill]
        df[f"{attr}_model_pred"] = o["preds"]
        df[f"{attr}_model_conf"] = o["confs"]
        df[f"{attr}_model_top2"] = o["top2"]
        report["attributes"][attr] = {
            "n_known_train": int(tr.sum()), "n_known_val": int(va.sum()),
            "method_choice": {k: {"train_acc_top1": v["train_acc"], "val_acc_top1": v["val_acc"]} for k, v in options.items()},
            "feature_used": method, "threshold": t, "threshold_source": "manual" if override is not None else "calibrated",
            "target_reached_on_train": reached, "train_precision_at_t": prec_tr, "train_coverage_at_t": cov_tr,
            "val_precision_at_t": float(o["correct"][va][sel_va].mean()) if sel_va.sum() else None,
            "val_coverage_at_t": float(sel_va.mean()),
            "val_curve": curve(o["confs"][va], o["correct"][va]),
            "train_class_precision_at_t": {c: {"precision": pr, "n": n_} for c, (pr, n_) in class_prec.items()},
            "blocked_classes": sorted(blocked),
            "val_per_class_precision_at_t": {
                c: float(o["correct"][va][sel_va & (np.array(o["preds"])[va] == c)].mean())
                for c in sorted(set(np.array(o["preds"])[va][sel_va]))},
            "blank_before": int((meta[col].fillna("").astype(str).str.strip() == "").sum()),
            "filled_by_source": df[src_col].value_counts().to_dict(),
            "blank_after": int((df[col] == "").sum()),
            "value_counts_after": df[col].value_counts().to_dict(),
        }

    out_cols = list(meta.columns) + ["color_query_raw", "subtype_source", "subtype_conf", "subtype_inconsistent",
                                     "color_source", "color_conf", "subtype_model_pred", "subtype_model_conf",
                                     "subtype_model_top2", "color_model_pred", "color_model_conf", "color_model_top2"]
    df[out_cols].to_parquet(OUT_PARQUET, index=False)
    report["n_subtype_inconsistent_original"] = int(df["subtype_inconsistent"].sum())
    save_json(report, "results/phase2.json")
    write_spotcheck(df, args.n_spotcheck)
    log_section(report)
    print(json.dumps({a: {k: v for k, v in r.items() if k in ("feature_used", "threshold", "train_precision_at_t",
                                                               "val_precision_at_t", "val_coverage_at_t",
                                                               "blank_before", "blank_after", "filled_by_source")}
                      for a, r in report["attributes"].items()}, indent=1))
    print(f"Wrote {OUT_PARQUET}, results/phase2.json, {SPOTCHECK_HTML}")


def log_section(r):
    lines = [f"## Phase 2 — metadata backfill ({r['created']})", "",
             "Fills blanks only, into `data/catalogue_metadata_backfilled.parquet` (production parquet untouched). "
             "Model threshold calibrated on train-split listings with known labels, target precision "
             f"{r['target_precision']}; reported on val-split known listings. Spot-check result pending (🙋).", "",
             "| attribute | blank before | blank after | filled: variant / title / model | model | threshold | model val precision @t | model val coverage @t |",
             "|---|---|---|---|---|---|---|---|"]
    blocked_note = []
    for a, x in r["attributes"].items():
        f = x["filled_by_source"]
        blocked_note.append(f"{a}: {', '.join(x['blocked_classes']) or 'none'}")
        lines.append(f"| {a} | {x['blank_before']} | {x['blank_after']} | {f.get('variant', 0)} / {f.get('title', 0)} / {f.get('probe', 0) + f.get('clip', 0)} | "
                     f"{x['feature_used']} | {x['threshold']:.2f} ({x['threshold_source']}) | "
                     f"{(x['val_precision_at_t'] or 0):.3f} | {x['val_coverage_at_t']:.3f} |")
    lines += ["", "Classes never filled by the model (train precision < min or < 10 samples): " + "; ".join(blocked_note)]
    lines += ["", "Variant colour strings → palette:", "", "| raw | palette |", "|---|---|"]
    lines += [f"| {k} | {v or '(blank → title/model)'} |" for k, v in r["variant_color_map"].items()]
    replace_log_section("## Phase 2 — metadata backfill", "\n".join(lines))


def write_spotcheck(df, n):
    # stratified by (attribute, source) so model fills are not drowned out by the many title fills
    strata = [(a, s) for a in ("subtype", "color") for s in ("title", "clip", "probe")
              if (df[f"{a}_source"] == s).any()]
    per = n // len(strata)
    picked = []
    for a, s in strata:
        pool = df[(df[f"{a}_source"] == s) & ~df.index.isin(picked)]
        picked += list(pool.sample(n=min(per, len(pool)), random_state=SEED).index)
    rest = df[(df["subtype_source"].isin(["title", "clip", "probe"]) | df["color_source"].isin(["title", "clip", "probe"]))
              & ~df.index.isin(picked)]
    picked += list(rest.sample(n=max(0, n - len(picked)), random_state=SEED).index)
    sample = df.loc[picked].sample(frac=1, random_state=SEED)
    emb_meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    imgs_of = emb_meta.groupby("catalogue_id")["image_path"].apply(list).to_dict()
    cards = []
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        others = [p for p in imgs_of.get(r["catalogue_id"], []) if p != r["primary_image"]][:2]
        thumbs = "".join(f'<img src="{thumb_b64(p, 170 if j == 0 else 90)}" class="{"big" if j == 0 else "small"}">'
                         for j, p in enumerate([r["primary_image"]] + others))

        def attr_block(attr, label, value, source, conf, top2):
            if source not in ("title", "clip", "probe"):
                return f'<div class="attr muted">{label}: {value or "—"} <small>({source or "blank"}, not checked)</small></div>'
            model = source in ("clip", "probe")
            c = f"{conf:.2f}" if model else "from title"
            return (f'<div class="attr" data-row="{i}" data-attr="{attr}" data-src="{source}">{label}: <b>{value}</b> '
                    f'<small>{source} · {c}{" · 2nd: " + top2 if model else ""}</small>'
                    f'<span class="btns"><button data-v="1">✓</button><button data-v="0">✗</button></span></div>')
        cards.append(f'<div class="card"><div class="imgs">{thumbs}</div><div class="info"><div class="n">#{i} · {r["category"]}</div>'
                     f'<div class="title">{r["title"]}</div>'
                     + attr_block("subtype", "Subtype", r["subtype"], r["subtype_source"], r["subtype_conf"], r["subtype_model_top2"])
                     + attr_block("color", "Colour", r["color_query"], r["color_source"], r["color_conf"], r["color_model_top2"])
                     + "</div></div>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Backfill spot-check</title><style>
:root{{--bg:#fdfaf6;--card:#fff;--ink:#2b2633;--muted:#8a8494;--ok:#cdeccf;--bad:#f6d0d0;--acc:#b7a3d9}}
body{{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:16px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid #e7e0ee;z-index:2}}
.bar b{{font-size:15px}} .bar button{{margin-left:8px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(420px,100%),1fr));gap:12px;margin-top:12px}}
.card{{background:var(--card);border-radius:12px;padding:10px;display:flex;flex-wrap:wrap;gap:10px;box-shadow:0 1px 3px #0001}}
.imgs{{display:flex;flex-direction:column;gap:4px}} .big{{width:170px;height:170px;object-fit:contain;border-radius:8px;background:#f4f1ee}}
.small{{width:83px;height:83px;object-fit:cover;border-radius:6px;margin-right:4px;display:inline-block}}
.imgs .small{{display:inline-block}} .info{{flex:1 1 200px;min-width:0;font-size:13px}} .n{{color:var(--muted);font-size:11px}} .title{{margin:4px 0 8px;font-weight:600}}
.attr{{padding:6px;border-radius:8px;margin:4px 0;background:#f6f3fa}} .attr small{{color:var(--muted)}} .muted{{color:var(--muted);background:none}}
.btns{{float:right}} .btns button{{border:1px solid #ccc;background:#fff;border-radius:6px;padding:2px 10px;cursor:pointer;font-size:14px}}
.attr.v1{{background:var(--ok)}} .attr.v0{{background:var(--bad)}} button{{font:inherit}}
</style></head><body>
<div class="bar"><b>Backfill spot-check</b> — mark each <i>checked</i> attribute ✓ (correct) or ✗ (wrong). Your marks save in this browser.
<div id="tally" style="margin-top:6px"></div><button id="copy">Copy summary</button><button id="reset">Reset</button></div>
<div class="grid">{''.join(cards)}</div>
<script>
const KEY='jl_phase2_spotcheck';let st={{}};try{{st=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(st))}}catch(e){{}}}}
const attrs=[...document.querySelectorAll('.attr[data-row]')];
function paint(){{attrs.forEach(a=>{{const k=a.dataset.row+'|'+a.dataset.attr;a.classList.remove('v0','v1');if(k in st)a.classList.add('v'+st[k])}});
 const t={{}};attrs.forEach(a=>{{for(const g of [a.dataset.attr, a.dataset.attr+' / '+a.dataset.src]){{t[g]=t[g]||{{ok:0,n:0,total:0}};t[g].total++;
 const k=a.dataset.row+'|'+a.dataset.attr;if(k in st){{t[g].n++;t[g].ok+=st[k]}}}}}});
 document.getElementById('tally').textContent=Object.entries(t).sort().map(([g,v])=>`${{g}}: ${{v.ok}}/${{v.n}} correct (${{v.total-v.n}} unmarked)`).join('   ·   ');
 return t}}
attrs.forEach(a=>a.querySelectorAll('button').forEach(b=>b.onclick=()=>{{st[a.dataset.row+'|'+a.dataset.attr]=+b.dataset.v;save();paint()}}));
document.getElementById('copy').onclick=()=>{{const t=paint();const s=Object.entries(t).sort().map(([g,v])=>`${{g}} ${{v.ok}}/${{v.n}}`).join(', ');
 const wrong=Object.entries(st).filter(([k,v])=>v===0).map(([k])=>'#'+k.replace('|',' ')).join(', ');
 navigator.clipboard.writeText(s+(wrong?' | wrong: '+wrong:'')).then(()=>alert('Copied:\\n'+s))}};
document.getElementById('reset').onclick=()=>{{if(confirm('Clear all marks?')){{st={{}};save();paint()}}}};
paint();
</script></body></html>"""
    with open(SPOTCHECK_HTML, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
