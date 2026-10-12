"""
Phase 8 — recommender behaviour on top of retrieval.

  mmr(results)            diversity re-rank (Maximal Marginal Relevance; config.MMR_LAMBDA_PREDICT / _SIMILAR per endpoint)
                          using the DINOv2 embedding of each result's matched catalogue image
  split_tiers(results)    "closest_matches" (score >= config.HIGH_MATCH_THRESHOLD) / "similar_items"
  out_of_scope_prob(clip) CLIP out-of-scope probability of a crop (prompts fixed a priori, Phase 8)
  similar_to(catalogue_id) "more like this": the listing's own stored image embeddings as the query,
                          per-listing max similarity, the listing itself excluded, same-category listings
                          first (its category is known), then MMR
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
import config

_state = {}


def _catalogue():
    if "emb" not in _state:
        emb = np.load(config.CATALOGUE_EMBEDDINGS_PATH).astype(np.float32)
        _state["emb"] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
        _state["fid_of_path"] = {p: i for i, p in enumerate(meta["image_path"])}
        cids = meta["catalogue_id"].to_numpy()
        _state["cids"], _state["cats"], _state["paths"] = cids, meta["category"].to_numpy(), meta["image_path"].to_numpy()
        starts = np.r_[0, np.where(cids[1:] != cids[:-1])[0] + 1]
        _state["starts"], _state["counts"] = starts, np.diff(np.r_[starts, len(cids)])
        _state["listing_pos"] = {c: i for i, c in enumerate(cids[starts])}
    return _state


def mmr(results, lam=None):
    lam = config.MMR_LAMBDA_PREDICT if lam is None else lam
    if lam >= 1.0 or len(results) < 3:
        return results
    st = _catalogue()
    fids = [st["fid_of_path"].get(r.get("image_path")) for r in results]
    if any(f is None for f in fids):
        return results
    V = st["emb"][fids]
    S = V @ V.T
    cand, chosen = list(range(len(results))), []
    while cand:
        if not chosen:
            best = max(cand, key=lambda i: results[i]["score"])
        else:
            best = max(cand, key=lambda i: lam * results[i]["score"] - (1 - lam) * S[i, chosen].max())
        chosen.append(best)
        cand.remove(best)
    return [results[i] for i in chosen]


def split_tiers(results):
    t = config.HIGH_MATCH_THRESHOLD
    return ([r for r in results if r["score"] >= t], [r for r in results if r["score"] < t])


def out_of_scope_prob(clip_emb):
    if "oos" not in _state:
        _state["in_text"] = np.load(os.path.join(config.CATEGORY_V2_DIR, "clip_text.npy"))
        _state["oos"] = np.load(os.path.join(config.OPENSET_DIR, "oos_text.npy"))
    z = 100.0 * np.concatenate([_state["in_text"] @ clip_emb, _state["oos"] @ clip_emb])
    z = np.exp(z - z.max())
    p = z / z.sum()
    return float(p[len(_state["in_text"]):].sum())


def similar_to(catalogue_id, top_k=20):
    """'More like this' for a catalogue listing. Returns (results, query_listing_category) or (None, None)."""
    st = _catalogue()
    if catalogue_id not in st["listing_pos"]:
        return None, None
    li = st["listing_pos"][catalogue_id]
    a, n = st["starts"][li], st["counts"][li]
    sims = (st["emb"] @ st["emb"][a:a + n].T).max(axis=1)        # best match to any image of the listing
    sims[a:a + n] = -np.inf                                        # never recommend the listing itself
    per_listing = np.maximum.reduceat(sims, st["starts"])
    best_img = [int(st["starts"][j] + np.argmax(sims[st["starts"][j]:st["starts"][j] + st["counts"][j]]))
                for j in range(len(st["starts"]))]
    top = np.argsort(-per_listing)[:top_k]
    results = [{"catalogue_id": st["cids"][st["starts"][j]], "category": st["cats"][st["starts"][j]],
                "image_path": st["paths"][best_img[j]], "score": float(per_listing[j])} for j in top]
    category = st["cats"][a]
    # the listing's category is known exactly: same-category listings first, scores stay plain cosine
    results.sort(key=lambda r: (r["category"] != category, -r["score"]))
    return mmr(results, config.MMR_LAMBDA_SIMILAR), category
