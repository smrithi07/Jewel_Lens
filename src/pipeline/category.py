"""
Phase 4 — fused category for one detected item (live path, behind config.CATEGORY_MODE).

Combines per-signal category probabilities with the weights chosen on the stumper
TUNE half (artifacts/category_v2/fusion.json, written by src/eval/category_fusion.py):
  det    : OWL-ViT label (smoothed one-hot)
  knn    : similarity-weighted category vote of the top-k retrieved listings
  clip   : CLIP image forward pass against precomputed text embeddings (no text model live)
  clf_v1 : the original classifier fed the un-normalised embedding it was trained on
  clf_v2 : retrained classifier on L2-normalised embeddings, temperature-scaled
Only the signals with a non-zero weight are computed, so unused models never load.
The detector label is one input, not the final authority.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.append(".")
import config

CATS = config.CATEGORIES
_state = None


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def load_category_models():
    global _state
    if _state is not None:
        return _state
    d = config.CATEGORY_V2_DIR
    fusion = json.load(open(os.path.join(d, "fusion.json")))
    weights = fusion["weights"] or {}
    st = {"fusion": fusion, "weights": weights}
    if "clf_v2" in weights:
        st["v2"] = pickle.load(open(os.path.join(d, "classifier.pkl"), "rb"))
    if "clf_v1" in weights:
        st["v1"] = pickle.load(open(config.CLASSIFIER_PATH, "rb"))
    if "clip" in weights:
        from src.catalogue.build_clip_embeddings import load_clip
        load_clip()
        st["clip_text"] = np.load(os.path.join(d, "clip_text.npy"))
    _state = st
    return st


def _ordered(model_dict, row):
    return np.array([row[list(model_dict["classes"]).index(c)] for c in CATS])


def signal_probs(crop, emb_norm, emb_raw, detector_category, neighbours, clip_emb=None):
    """neighbours: retrieval results (dicts with 'category' and 'score'), best first.
    clip_emb: optional precomputed CLIP image embedding of the crop (offline evaluation)."""
    st = load_category_models()
    w, f = st["weights"], st["fusion"]
    out = {}
    if "det" in w:
        eps = f["det_eps"]
        p = np.full(3, eps / 3)
        p[CATS.index(detector_category)] += 1 - eps
        out["det"] = p
    if "knn" in w:
        nb = neighbours[: f["knn"]["k"]]
        s = np.array([r["score"] for r in nb])
        wt = np.exp((s - s.max()) / f["knn"]["T"]) if len(nb) else np.array([])
        p = np.zeros(3)
        for r, wi in zip(nb, wt):
            p[CATS.index(r["category"])] += wi
        out["knn"] = (p + 0.01) / (p.sum() + 0.03)
    if "clip" in w:
        if clip_emb is None:
            from src.catalogue.build_clip_embeddings import clip_image_embed
            clip_emb = clip_image_embed([crop])[0]
        out["clip"] = _softmax(100.0 * clip_emb @ st["clip_text"].T)
        out["_clip_emb"] = clip_emb
    if "clf_v1" in w:
        out["clf_v1"] = _ordered(st["v1"], st["v1"]["model"].predict_proba(emb_raw.reshape(1, -1))[0])
    if "clf_v2" in w:
        v2 = st["v2"]
        out["clf_v2"] = _softmax(_ordered(v2, v2["model"].decision_function(emb_norm.reshape(1, -1))[0]) / v2["temperature"])
    return out


def categorize(crop, emb_norm, emb_raw, detector_category, neighbours, clip_emb=None):
    """-> {category, confidence, probs{cat: p}, signals{name: {cat: p}}}"""
    st = load_category_models()
    sig = signal_probs(crop, emb_norm, emb_raw, detector_category, neighbours, clip_emb)
    clip_vec = sig.pop("_clip_emb", None)
    tot = sum(st["weights"].values())
    p = sum(st["weights"][s] * sig[s] for s in sig) / tot
    i = int(np.argmax(p))
    return {"category": CATS[i], "confidence": float(p[i]),
            "probs": {c: float(v) for c, v in zip(CATS, p)},
            "signals": {s: {c: float(v) for c, v in zip(CATS, q)} for s, q in sig.items()},
            "clip_emb": clip_vec}
