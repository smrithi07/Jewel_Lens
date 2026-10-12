"""
Phase 4 — compare category signals alone and fused; choose on TUNE, report HOLDOUT.

Signals (each a probability vector over config.CATEGORIES, per detected item)
  det      : OWL-ViT label, smoothed one-hot ((1-eps) on the label, eps spread)
  knn      : similarity-weighted category vote of the top-k retrieved listings
             (softmax(score / T)); k, T chosen on tune
  clip     : CLIP zero-shot on the crop. The prompt set is chosen OFFLINE on train-split
             catalogue images (stored CLIP embeddings); text embeddings are precomputed,
             so live inference is one image forward pass
  clf_v1   : the existing classifier, fed the un-normalised embedding it was trained on
             (i.e. the Phase 0 input bug fixed, nothing retrained)
  clf_v2   : classifier retrained on train-split clean + augmented + detector-crop
             features (src/catalogue/train_category_v2.py), temperature-scaled on val
  legacy   : the classifier exactly as deployed (L2-normalised input) — the baseline

Fusion: weighted average of the signal probabilities, weights on a coarse grid
{0, 0.5, 1, 2}; chosen by tune item accuracy, ties -> higher tune photo F1 ->
fewer signals. Also reported: logistic-regression stacking with leave-one-photo-out CV.

Ground truth per item: folder name for earring/necklace/bracelet photos (all detected
items in such a photo are taken as that category). Mixed photos enter only through the
item-category F1 against the user's per-category counts (labels.csv).

Usage: python src/eval/category_fusion.py
  -> results/phase4.json, data/eval_own/phase4_detections.json, artifacts/category_v2/fusion.json, LOG.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import copy
import itertools
import json
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(".")
import config
from src.eval.common import (SPLITS_PATH, STUMPER_SPLIT_PATH, bootstrap_ci, fmt_ci, load_json, save_json,
                             replace_log_section)
from src.eval import own_eval
from src.catalogue.build_clip_embeddings import clip_text_embed, OUT_DIR as CLIP_DIR

CATS = config.CATEGORIES
ITEMS = "data/eval_own/phase4_items.npz"
V2_DIR = "artifacts/category_v2"
SIGNALS = ["det", "knn", "clip", "clf_v1", "clf_v2"]
DET_EPS = 0.2

PROMPT_SETS = {
    "names": {"earring": ["an earring"], "necklace": ["a necklace"], "bracelet": ["a bracelet"]},
    "names+subtypes": {
        "earring": ["an earring", "a pair of earrings", "a jhumka earring", "a hoop earring", "a stud earring",
                    "a drop earring", "a chandbali earring", "an ear cuff"],
        "necklace": ["a necklace", "a pendant necklace", "a choker necklace", "a chain necklace",
                     "a layered necklace", "a beaded necklace", "a mangalsutra"],
        "bracelet": ["a bracelet", "a bangle", "a cuff bracelet", "a chain bracelet", "a kada",
                     "a beaded bracelet", "a tennis bracelet", "a charm bracelet"]},
}
TEMPLATE_SETS = {
    "plain": ["a photo of {}."],
    "ensemble": ["a photo of {}.", "a close-up photo of {}.", "a product photo of {}.",
                 "a photo of {} being worn.", "a phone photo of {} on a table."],
}


# ------------------------------------------------------------------ signals

def det_probs(labels):
    p = np.full((len(labels), 3), DET_EPS / 3)
    for i, l in enumerate(labels):
        p[i, CATS.index(l)] += 1 - DET_EPS
    return p


def knn_probs(knn, k, T):
    p = np.zeros((len(knn), 3))
    for i, nb in enumerate(knn):
        nb = nb[:k]
        s = np.array([x[1] for x in nb])
        w = np.exp((s - s.max()) / T)
        for (c, _, _), wi in zip(nb, w):
            p[i, CATS.index(c)] += wi
        p[i] = (p[i] + 0.01) / (p[i].sum() + 0.03)
    return p


def clip_text_matrix(prompt_set, template_set):
    rows = []
    for c in CATS:
        e = clip_text_embed([t.format(n) for n in PROMPT_SETS[prompt_set][c] for t in TEMPLATE_SETS[template_set]]).mean(0)
        rows.append(e / np.linalg.norm(e))
    return np.stack(rows)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def choose_clip_prompts():
    """Offline: pick the prompt/template set by accuracy on train-split, non-set catalogue images."""
    listings = load_json(SPLITS_PATH)["listings"]
    emb = np.load(os.path.join(CLIP_DIR, "embeddings.npy"))
    meta = pd.read_parquet(os.path.join(CLIP_DIR, "meta.parquet"))
    m = meta["catalogue_id"].map(lambda c: listings[c]["split"] == "train" and not listings[c]["is_set"]).to_numpy()
    y = meta["category"].to_numpy()[m]
    scores = {}
    for ps, ts in itertools.product(PROMPT_SETS, TEMPLATE_SETS):
        T = clip_text_matrix(ps, ts)
        pred = np.array(CATS)[(emb[m] @ T.T).argmax(1)]
        scores[f"{ps}|{ts}"] = float((pred == y).mean())
    best = max(scores, key=scores.get)
    return best, scores


def ece(probs, y_idx, n_bins=10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y_idx
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


# ------------------------------------------------------------------ evaluation helpers

def item_metrics(probs, rows, idx):
    """Accuracy over single-category-folder items in idx; returns (acc, per-item correct list)."""
    sel = [i for i in idx if rows[i]["folder"] in CATS]
    correct = [float(CATS[int(np.argmax(probs[i]))] == rows[i]["folder"]) for i in sel]
    return (float(np.mean(correct)) if correct else None), correct


def confusion(probs, rows, idx):
    cm = {t: {p: 0 for p in CATS} for t in CATS}
    for i in idx:
        if rows[i]["folder"] in CATS:
            cm[rows[i]["folder"]][CATS[int(np.argmax(probs[i]))]] += 1
    return cm


def with_field(dets, rows, probs, field):
    d = copy.deepcopy(dets)
    lookup = {(r["photo"], r["item_index"]): i for i, r in enumerate(rows)}
    for photo, v in d.items():
        for it in v["items"]:
            i = lookup[(photo, it["item_index"])]
            it[field] = CATS[int(np.argmax(probs[i]))]
            it[field.replace("_category", "_confidence")] = float(np.max(probs[i]))
    return d


# ------------------------------------------------------------------ main

def main():
    z = np.load(ITEMS)
    rows, knn = json.loads(str(z["rows"])), json.loads(str(z["knn"]))
    dino_raw, clip_emb = z["dino"], z["clip"]
    dino_n = dino_raw / np.linalg.norm(dino_raw, axis=1, keepdims=True)
    tune = [i for i, r in enumerate(rows) if r["half"] == "tune" and r["folder"] != "negatives"]
    hold = [i for i, r in enumerate(rows) if r["half"] == "holdout" and r["folder"] != "negatives"]

    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    dets = load_json(own_eval.BASELINE_DETS)
    labels = own_eval.load_labels()
    report = {"created": datetime.now().isoformat(timespec="seconds"), "n_items": {"tune": len(tune), "holdout": len(hold)}}

    probs = {}
    # legacy (as deployed) and clf_v1 (input fixed)
    v1 = pickle.load(open(config.CLASSIFIER_PATH, "rb"))
    order = [list(v1["classes"]).index(c) for c in CATS]
    probs["legacy"] = v1["model"].predict_proba(dino_n)[:, order]
    probs["clf_v1"] = v1["model"].predict_proba(dino_raw)[:, order]
    probs["det"] = det_probs([r["detector_category"] for r in rows])

    # kNN: k and T chosen on tune
    grid = {(k, T): item_metrics(knn_probs(knn, k, T), rows, tune)[0] for k in (5, 10, 20) for T in (0.02, 0.05, 1.0)}
    k_best, T_best = max(grid, key=lambda kt: (grid[kt], -kt[0]))
    probs["knn"] = knn_probs(knn, k_best, T_best)
    report["knn"] = {"k": k_best, "T": T_best, "tune_grid": {f"k={k},T={T}": v for (k, T), v in grid.items()}}

    # CLIP: prompts chosen offline on train catalogue images
    best_prompts, prompt_scores = choose_clip_prompts()
    ps, ts = best_prompts.split("|")
    text = clip_text_matrix(ps, ts)
    probs["clip"] = softmax(100.0 * clip_emb @ text.T)
    report["clip"] = {"prompt_choice_train_catalogue_acc": prompt_scores, "chosen": best_prompts}
    np.save(os.path.join(V2_DIR, "clip_text.npy"), text)

    # clf_v2 if trained
    v2_path = os.path.join(V2_DIR, "classifier.pkl")
    if os.path.exists(v2_path):
        v2 = pickle.load(open(v2_path, "rb"))
        o2 = [list(v2["classes"]).index(c) for c in CATS]
        probs["clf_v2"] = softmax(v2["model"].decision_function(dino_n)[:, o2] / v2["temperature"])
        report["clf_v2"] = {k: v for k, v in v2.items() if k != "model"}
    signals = [s for s in SIGNALS if s in probs]

    # ECE on stumper items (tune + holdout single-category items)
    lab_idx = [i for i in tune + hold if rows[i]["folder"] in CATS]
    y_idx = np.array([CATS.index(rows[i]["folder"]) for i in lab_idx])
    report["ece_stumper"] = {s: ece(probs[s][lab_idx], y_idx) for s in ["legacy"] + signals if s != "det"}

    # ---- alone + all equal-weight subsets
    def fuse(w):
        tot = sum(w.values())
        return sum(w[s] * probs[s] for s in w if w[s] > 0) / tot

    candidates = {}
    for s in ["legacy"] + signals:
        candidates[s] = {s: 1.0}
    for r_ in range(2, len(signals) + 1):
        for combo in itertools.combinations(signals, r_):
            candidates["+".join(combo)] = {s: 1.0 for s in combo}
    # weighted grid
    best_grid, best_key = None, None
    for ws in itertools.product([0, 0.5, 1, 2], repeat=len(signals)):
        if sum(ws) == 0:
            continue
        w = dict(zip(signals, ws))
        acc = item_metrics(fuse(w), rows, tune)[0]
        key = (acc, -sum(1 for x in ws if x > 0), -sum(ws))
        if best_key is None or key > best_key:
            best_key, best_grid = key, w
    candidates["fused_grid"] = {s: v for s, v in best_grid.items() if v > 0}

    # LR stacking with leave-one-photo-out CV on tune (honest estimate), then fit on all tune
    from sklearn.linear_model import LogisticRegression
    Xs = np.hstack([np.log(probs[s] + 1e-4) for s in signals])
    tune_lab = [i for i in tune if rows[i]["folder"] in CATS]
    yt = np.array([rows[i]["folder"] for i in tune_lab])
    photos_t = np.array([rows[i]["photo"] for i in tune_lab])
    loo_correct = []
    for p in np.unique(photos_t):
        tr, te = photos_t != p, photos_t == p
        if len(set(yt[tr])) < 3:
            continue
        lr = LogisticRegression(C=0.3, max_iter=2000).fit(Xs[tune_lab][tr], yt[tr])
        loo_correct += list(lr.predict(Xs[tune_lab][te]) == yt[te])
    lr = LogisticRegression(C=0.3, max_iter=2000).fit(Xs[tune_lab], yt)
    probs["stack_lr"] = lr.predict_proba(Xs)[:, [list(lr.classes_).index(c) for c in CATS]]
    report["stack_lr_tune_loo_acc"] = float(np.mean(loo_correct))

    # ---- score every candidate
    table = {}
    for name, w in list(candidates.items()) + [("stack_lr", None)]:
        P = probs["stack_lr"] if name == "stack_lr" else fuse(w)
        field = "c_" + name.replace("+", "_") + "_category"
        dd = with_field(dets, rows, P, field)
        r = {"weights": w}
        for half, idx in (("tune", tune), ("holdout", hold)):
            acc, corr = item_metrics(P, rows, idx)
            ev = own_eval.evaluate(dd, split, labels, [], half, field)
            r[half] = {"item_acc": bootstrap_ci(corr), "photo_acc_first": ev["category_acc_first"],
                       "item_category_f1": ev["item_category_f1"], "confusion": confusion(P, rows, idx),
                       "by_tag": {t: {"n": v["n"], "f1": v["item_category_f1"]["mean"], "acc_first": v["category_acc_first"]["mean"]}
                                  for t, v in ev["by_tag"].items()}}
        table[name] = r
    report["candidates"] = table

    # ---- winner, chosen on tune only. Honest tune scores: stack_lr uses its leave-one-photo-out
    # accuracy (its in-sample tune accuracy is fitted). fused_grid was picked among ~1000 weightings,
    # so it must beat the best equal-weight candidate by more than one tune item to be preferred.
    n_tune_items = len([i for i in tune if rows[i]["folder"] in CATS])

    def tune_acc(n):
        return report["stack_lr_tune_loo_acc"] if n == "stack_lr" else table[n]["tune"]["item_acc"]["mean"]

    def key(n):
        return (round(tune_acc(n), 4), round(table[n]["tune"]["item_category_f1"]["mean"] or 0, 4),
                -(len(candidates.get(n) or {}) if n != "stack_lr" else 99))
    eligible = [n for n in table if n != "legacy"]
    best_equal = max([n for n in eligible if n not in ("fused_grid", "stack_lr")], key=key)
    winner = max(eligible, key=key)
    if winner == "fused_grid" and tune_acc("fused_grid") - tune_acc(best_equal) <= 1.0 / n_tune_items + 1e-9:
        winner = best_equal
    report["selection"] = {"honest_tune_acc": {n: tune_acc(n) for n in table}, "best_equal_weight": best_equal,
                           "rule": "max honest tune item acc; grid needs > 1 item margin over best equal-weight"}
    report["winner"] = winner
    win_w = candidates.get(winner)
    P_win = probs["stack_lr"] if winner == "stack_lr" else fuse(win_w)
    final = with_field(dets, rows, P_win, "fused_category")
    for s in ["legacy"] + signals:
        final = with_field(final, rows, probs[s], f"{s}_category")
    save_json(final, "data/eval_own/phase4_detections.json")
    save_json({"winner": winner, "weights": win_w, "signals": signals, "det_eps": DET_EPS,
               "knn": {"k": k_best, "T": T_best}, "clip_prompts": best_prompts,
               "note": "weights of a weighted average of per-signal category probabilities"},
              os.path.join(V2_DIR, "fusion.json"))
    save_json(report, "results/phase4.json")
    replace_log_section("## Phase 4 — category identification", markdown(report, table, winner, signals))
    print(markdown(report, table, winner, signals))


def markdown(report, table, winner, signals):
    def row(n):
        t, h = table[n]["tune"], table[n]["holdout"]
        return (f"| {n}{' **(winner)**' if n == winner else ''} | {t['item_acc']['mean']:.3f} | {fmt_ci(h['item_acc'])} | "
                f"{(t['photo_acc_first']['mean'] or 0):.3f} | {fmt_ci(h['photo_acc_first'])} | "
                f"{(t['item_category_f1']['mean'] or 0):.3f} | {fmt_ci(h['item_category_f1'])} |")
    singles = ["legacy"] + signals
    best_combos = sorted([n for n in table if n not in singles and n not in (winner, "stack_lr")],
                         key=lambda n: -table[n]["tune"]["item_acc"]["mean"])[:6]
    lines = [f"## Phase 4 — category identification ({report['created']})", "",
             f"Same OWL-ViT boxes as the baseline. Items: tune {report['n_items']['tune']}, holdout {report['n_items']['holdout']} "
             "(item truth = folder for single-category photos; mixed photos count via F1 against labelled counts). "
             "Chosen on **tune**, reported on **holdout** (95% bootstrap CIs). "
             f"kNN k={report['knn']['k']}, T={report['knn']['T']} (chosen on tune). CLIP prompts `{report['clip']['chosen']}` "
             "(chosen offline on train catalogue images). "
             f"LR stacking leave-one-photo-out acc on tune: {report['stack_lr_tune_loo_acc']:.3f}.", "",
             "| signal(s) | tune item acc | holdout item acc | tune photo acc | holdout photo acc (first item) | tune item-cat F1 | holdout item-cat F1 |",
             "|---|---|---|---|---|---|---|"]
    lines += [row(n) for n in singles] + [row(n) for n in best_combos] + [row(winner)]
    if winner != "stack_lr":
        lines.append(row("stack_lr"))
    lines += ["", f"Winner weights: `{table[winner]['weights']}`", "",
              "ECE on stumper items (lower = better calibrated): " +
              ", ".join(f"{k} {v:.3f}" for k, v in report["ece_stumper"].items()), ""]
    for n in ("legacy", winner):
        cm = table[n]["holdout"]["confusion"]
        lines += [f"Holdout confusion, {n} (rows = true; cols = earring / necklace / bracelet):", "",
                  "| true | earring | necklace | bracelet |", "|---|---|---|---|"]
        lines += [f"| {t} | " + " | ".join(str(cm[t][p]) for p in CATS) + " |" for t in CATS] + [""]
    lines += ["Per-tag holdout (item-category F1 / first-item acc):", "", "| tag | n | legacy | det | winner |", "|---|---|---|---|---|"]
    for t, v in table[winner]["holdout"]["by_tag"].items():
        def f(n):
            x = table[n]["holdout"]["by_tag"].get(t, {})
            return f"{(x.get('f1') or 0):.2f} / {(x.get('acc_first') or 0):.2f}"
        lines.append(f"| {t} | {v['n']} | {f('legacy')} | {f('det')} | {f(winner)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
