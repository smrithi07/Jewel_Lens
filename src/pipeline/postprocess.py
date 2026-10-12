"""
Phase 5 — turn raw detector boxes into item boxes. Every rule is a toggle in one
config dict, so the benchmark can ablate them and the live pipeline uses the same code.

Query sets (per category; "agnostic" phrases score jewellery in general):
  current  : the three v1 templates (config.DETECTION_PROMPT_TEMPLATES)
  ensemble : several phrasings per category; a box's category score is the mean
             (or max) of its phrasings' scores

Steps (in order)
  1. score     : category scores from the query set; objectness = max category score
                 (agnostic_queries: also the generic "jewellery" phrases)
  2. threshold : objectness >= score_thr, box area >= min_area_frac of the image
  3. nms       : greedy NMS at nms_iou; all boxes compete (as torchvision.ops.nms in v1)
                 unless nms_per_class, then only boxes with the same top category
       + pair_protect: a box is NOT suppressed by a higher one when they look like a
         symmetric pair (similar size, both confident, neither mostly inside the other,
         side by side) — two overlapping earrings
  4. group_drop: drop a box that contains >= 2 other kept boxes (a box around a whole pair/set)
  5. chain_merge: merge two necklace/bracelet boxes when the union is much more elongated
                 than either part and the gap between them is small (a chain split in two)
  (tiles)        optional SAHI-style tiled pass: small items (earrings inside a necklace's loop)
                 get boxes the full-image pass misses at OWL-ViT's 32-px patch resolution
  (verify_sim)   applied by the caller after embedding: drop boxes whose crop's top-1 catalogue
                 similarity < verify_sim (the highest-scoring box is always kept)
Category of the final item comes from Phase 4 (src/pipeline/category.py); the detector's
own top category is kept as `detector_category`.
"""

import sys

import numpy as np

sys.path.append(".")
import config

CATS = config.CATEGORIES
QUERY_SETS = {
    "current": {c: [config.DETECTION_PROMPT_TEMPLATES[c]] for c in CATS},
    "ensemble": {
        "earring": ["a single earring", "an earring", "a jhumka earring", "a hoop earring", "a stud earring",
                    "a drop earring"],
        "necklace": ["a necklace worn around the neck", "a necklace", "a pendant necklace", "a chain necklace",
                     "a choker necklace"],
        "bracelet": ["a bracelet or bangle worn on the wrist", "a bracelet", "a bangle", "a chain bracelet",
                     "a cuff bracelet"],
        "agnostic": ["a piece of jewellery", "jewelry"],
    },
}
ALL_QUERIES = list(dict.fromkeys(q for s in QUERY_SETS.values() for qs in s.values() for q in qs))

# DEFAULT reproduces the v1 production detector post-processing
DEFAULT = {"query_set": "current", "agg": "max", "agnostic_queries": False, "nms_per_class": False, "score_thr": 0.1,
           "tiles": False, "tile_score_thr": None, "verify_sim": None,
           "min_area_frac": 0.0, "nms_iou": 0.3, "pair_protect": False, "group_drop": False,
           "chain_merge": False}


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def category_scores(det, queries, cfg):
    """det['scores'] aligned with `queries` -> ({cat: score}, agnostic score)."""
    qs = QUERY_SETS[cfg["query_set"]]
    idx = {q: i for i, q in enumerate(queries)}
    agg = np.mean if cfg["agg"] == "mean" else np.max
    cs = {c: float(agg([det["scores"][idx[q]] for q in qs[c]])) for c in CATS}
    ag = float(max(det["scores"][idx[q]] for q in qs.get("agnostic", []))) if "agnostic" in qs else 0.0
    return cs, ag


def _looks_like_pair(a, b, sa, sb, top):
    small, big = sorted([_area(a["box"]), _area(b["box"])])
    if big == 0 or small / big < 0.6:                       # similar size
        return False
    if min(sa, sb) < 0.6 * top:                             # both confident relative to the best box
        return False
    if _inter(a["box"], b["box"]) / small > 0.8:            # neither mostly inside the other
        return False
    wa, wb = a["box"][2] - a["box"][0], b["box"][2] - b["box"][0]
    dx = abs((a["box"][0] + a["box"][2]) - (b["box"][0] + b["box"][2])) / 2
    return dx > 0.3 * min(wa, wb)                           # side by side


def _chain_merge(items, W, H):
    def aspect(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return max(w, h) / max(1e-6, min(w, h))

    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["detector_category"] not in ("necklace", "bracelet") or b["detector_category"] != a["detector_category"]:
                    continue
                gx = max(0.0, max(a["box"][0], b["box"][0]) - min(a["box"][2], b["box"][2]))
                gy = max(0.0, max(a["box"][1], b["box"][1]) - min(a["box"][3], b["box"][3]))
                size = min(max(a["box"][2] - a["box"][0], a["box"][3] - a["box"][1]),
                           max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]))
                u = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]),
                     max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                if np.hypot(gx, gy) <= 0.25 * size and aspect(u) >= 1.5 * max(aspect(a["box"]), aspect(b["box"])):
                    items[i] = {**a, "box": u, "score": max(a["score"], b["score"]), "merged": True}
                    items.pop(j)
                    merged = True
                    break
            if merged:
                break
    return items


def postprocess(raw, queries, size, cfg=None, tile_raw=None):
    """raw: list of {'box', 'scores'} from detectors.raw_detect; size=(W, H) -> list of items.
    tile_raw: optional boxes from a tiled pass (already in full-image coordinates, with a 'cut' flag);
    used when cfg['tiles'] — boxes cut by an interior tile edge are dropped (SAHI-style)."""
    cfg = {**DEFAULT, **(cfg or {})}
    W, H = size
    cands = []
    pool = [(d, cfg["score_thr"]) for d in raw]
    if cfg["tiles"] and tile_raw:
        thr_t = cfg["tile_score_thr"] if cfg["tile_score_thr"] is not None else cfg["score_thr"]
        pool += [(d, thr_t) for d in tile_raw if not d.get("cut")]
    for d, thr in pool:
        cs, ag = category_scores(d, queries, cfg)
        top_cat = max(cs, key=cs.get)
        obj = max(cs[top_cat], ag) if cfg["agnostic_queries"] else cs[top_cat]
        if obj < thr or _area(d["box"]) < cfg["min_area_frac"] * W * H:
            continue
        cands.append({"box": d["box"], "score": obj, "detector_category": top_cat, "category_scores": cs})
    cands.sort(key=lambda x: -x["score"])
    top = cands[0]["score"] if cands else 0.0

    kept = []
    for c in cands:
        suppressed = False
        for k in kept:
            if cfg["nms_per_class"] and k["detector_category"] != c["detector_category"]:
                continue
            if iou(c["box"], k["box"]) > cfg["nms_iou"]:
                if cfg["pair_protect"] and _looks_like_pair(c, k, c["score"], k["score"], top):
                    continue
                suppressed = True
                break
        if not suppressed:
            kept.append(c)

    if cfg["group_drop"]:
        def inside(small, big):
            return _inter(small["box"], big["box"]) / max(1e-6, _area(small["box"])) >= 0.85
        kept = [b for b in kept if sum(1 for o in kept if o is not b and _area(o["box"]) < _area(b["box"]) and inside(o, b)) < 2]

    if cfg["chain_merge"]:
        kept = _chain_merge(kept, W, H)

    return [{**k, "box": [float(v) for v in k["box"]], "score": float(k["score"])} for k in kept]
