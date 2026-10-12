"""
Phase 4 — retrain the category classifier for real crops.

Fixes the two v1 problems: (1) it is trained on L2-NORMALISED DINOv2 embeddings,
exactly what src/pipeline/embed.embed_image returns at inference (v1 was trained
on raw vectors and fed normalised ones, so it always said "earring"); (2) it sees
augmented views and detector crops, not just clean catalogue photos.

Data: artifacts/category_v2/features_{train,val}.npz (build_category_data.py),
catalogue train/val listings only, set listings excluded. No stumper photos.

Selection (no stumper data used):
  - training-source set {clean} / {clean+aug} / {clean+aug+detcrop}: chosen by val
    accuracy on the hard sources (aug + detcrop)
  - C: GroupKFold(5) by listing on train, balanced accuracy
  - balanced class weights (removes the earring prior); temperature scaling on val (NLL)
Reports ECE before/after temperature on val, intercepts, and per-source val accuracy.
Stumper tune/holdout numbers are reported for information only (not used to choose).

Usage: python src/catalogue/train_category_v2.py -> artifacts/category_v2/classifier.pkl, results/phase4_classifier.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

sys.path.append(".")
import config
from src.eval.common import save_json

V2 = "artifacts/category_v2"
CATS = config.CATEGORIES
SOURCE_SETS = {"clean": ["clean"], "clean+aug": ["clean", "aug"], "clean+aug+detcrop": ["clean", "aug", "detcrop"]}
C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]


def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(p, y_idx, n_bins=10):
    conf, correct = p.max(1), p.argmax(1) == y_idx
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def fit_temperature(logits, y_idx):
    best = (np.inf, 1.0)
    for T in np.round(np.arange(0.2, 5.01, 0.05), 2):
        p = softmax(logits / T)
        nll = -np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12))
        best = min(best, (nll, float(T)))
    return best[1]


def main():
    tr, va = np.load(f"{V2}/features_train.npz"), np.load(f"{V2}/features_val.npz")
    Xtr, ytr, gtr, str_ = l2n(tr["X"]), tr["y"], tr["catalogue_id"], tr["source"]
    Xva, yva, sva = l2n(va["X"]), va["y"], va["source"]
    yva_idx = np.array([CATS.index(c) for c in yva])
    hard_va = np.isin(sva, ["aug", "detcrop"])
    print(f"train {Xtr.shape} {dict(zip(*np.unique(str_, return_counts=True)))} | val {Xva.shape}")

    report = {"source_sets": {}}
    best = None
    for name, srcs in SOURCE_SETS.items():
        m = np.isin(str_, srcs)
        t0 = time.time()
        cv = {}
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            cv[C] = float(cross_val_score(clf, Xtr[m], ytr[m], groups=gtr[m], cv=GroupKFold(5),
                                          scoring="balanced_accuracy").mean())
        C = max(cv, key=cv.get)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr[m], ytr[m])
        order = [list(clf.classes_).index(c) for c in CATS]
        pred = np.array(CATS)[clf.decision_function(Xva)[:, order].argmax(1)]
        per_src = {s: float((pred[sva == s] == yva[sva == s]).mean()) for s in np.unique(sva)}
        hard = float((pred[hard_va] == yva[hard_va]).mean())
        report["source_sets"][name] = {"cv_balanced_acc": cv, "C": C, "val_acc_by_source": per_src,
                                       "val_acc_hard": hard, "intercepts": dict(zip(CATS, clf.intercept_[order].round(3).tolist()))}
        print(f"  {name:18s} C={C:<5} cv={cv[C]:.3f} val={per_src} hard={hard:.3f} ({time.time() - t0:.0f}s)")
        if best is None or hard > best[0]:
            best = (hard, name, clf, order, C)

    _, name, clf, order, C = best
    logits = clf.decision_function(Xva)[:, order]
    T = fit_temperature(logits, yva_idx)
    report["chosen"] = {"source_set": name, "C": C, "temperature": T,
                        "val_ece_before": ece(softmax(logits), yva_idx), "val_ece_after": ece(softmax(logits / T), yva_idx),
                        "val_acc_all": float((logits.argmax(1) == yva_idx).mean())}

    # v1 classifier on the same val features, for "before" calibration
    v1 = pickle.load(open(config.CLASSIFIER_PATH, "rb"))
    o1 = [list(v1["classes"]).index(c) for c in CATS]
    for tag, X in (("legacy_normalised_input", Xva), ("v1_raw_input", va["X"])):
        p = v1["model"].predict_proba(X)[:, o1]
        report[f"v1_{tag}"] = {"val_acc": float((p.argmax(1) == yva_idx).mean()), "val_ece": ece(p, yva_idx)}

    with open(f"{V2}/classifier.pkl", "wb") as f:
        pickle.dump({"model": clf, "classes": list(clf.classes_), "C": C, "temperature": T, "input": "l2_normalised_dinov2_cls",
                     "train_sources": SOURCE_SETS[name], "trained_on": "train-split non-set listings"}, f)
    save_json(report, "results/phase4_classifier.json")
    print(json.dumps(report["chosen"], indent=1))
    print({k: v for k, v in report.items() if k.startswith("v1_")})


if __name__ == "__main__":
    main()
