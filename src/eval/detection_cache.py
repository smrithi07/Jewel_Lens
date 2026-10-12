"""
Phase 5 — run each detector ONCE per stumper photo at a low score floor and cache
every raw box with its score for every query, so thresholds / NMS / prompt sets /
post-processing rules can be swept offline in seconds.

OWL-ViT and OWLv2 score all queries in one pass (union of every query set).
Grounding DINO grounds phrases inside one caption, so each query set is its own pass
(a long caption is not the same as several short ones); its latency is per pass.

Usage: python src/eval/detection_cache.py --detector owlv2   [--half all]
  -> data/eval_own/phase5_raw_<detector>.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import time

sys.path.append(".")
from src.eval.common import STUMPER_SPLIT_PATH, Progress, load_json, save_json
from src.pipeline.detectors import raw_detect, load
from src.pipeline.postprocess import QUERY_SETS, ALL_QUERIES
from src.pipeline.preprocess import load_and_preprocess


TILE_FRAC = 0.6


def tile_boxes(W, H):
    tw, th = TILE_FRAC * W, TILE_FRAC * H
    xs, ys = [0, W - tw], [0, H - th]
    tiles = [(x, y) for y in ys for x in xs] + [((W - tw) / 2, (H - th) / 2)]
    return [(x, y, x + tw, y + th) for x, y in tiles]


def tiled_pass(detector, img, passes):
    """Detect on each tile (upsampled by the processor), map boxes back, and flag boxes that
    touch an interior tile edge (likely a cut-off piece of a larger item)."""
    W, H = img.size
    queries = passes["all"] if "all" in passes else passes["ensemble"]
    dets, t0 = [], time.perf_counter()
    for (x0, y0, x1, y1) in tile_boxes(W, H):
        tile = img.crop((x0, y0, x1, y1))
        margin = 0.02 * max(x1 - x0, y1 - y0)
        for d in raw_detect(detector, tile, queries):
            b = d["box"]
            cut = ((b[0] < margin and x0 > 0) or (b[1] < margin and y0 > 0) or
                   (b[2] > (x1 - x0) - margin and x1 < W) or (b[3] > (y1 - y0) - margin and y1 < H))
            dets.append({"box": [round(b[0] + x0, 1), round(b[1] + y0, 1), round(b[2] + x0, 1), round(b[3] + y0, 1)],
                         "scores": d["scores"], "cut": bool(cut)})
    return {"queries": queries, "dets": dets, "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["owlvit", "owlv2", "gdino"])
    ap.add_argument("--half", default="all", choices=["tune", "holdout", "all"])
    ap.add_argument("--tiles", action="store_true", help="add a tiled pass (2x2 + centre, 60%% tiles) to an existing cache")
    args = ap.parse_args()
    out_path = f"data/eval_own/phase5_raw_{args.detector}.json"
    cache = load_json(out_path) if os.path.exists(out_path) else {}

    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    photos = [p for p, v in split.items() if args.half == "all" or v["half"] == args.half]
    load(args.detector)
    if args.detector == "gdino":
        passes = {name: list(dict.fromkeys(q for qs in s.values() for q in qs)) for name, s in QUERY_SETS.items()}
    else:
        passes = {"all": ALL_QUERIES}

    prog = Progress(len(photos), args.detector)
    for i, p in enumerate(photos):
        if p in cache and (not args.tiles or "tiles" in cache[p]["passes"]):
            prog.update(i + 1)
            continue
        img = load_and_preprocess(p)
        if args.tiles and p in cache:
            cache[p]["passes"]["tiles"] = tiled_pass(args.detector, img, passes)
            if (i + 1) % 10 == 0:
                save_json(cache, out_path)
            prog.update(i + 1)
            continue
        entry = {"size": list(img.size), "passes": {}}
        for name, queries in passes.items():
            t0 = time.perf_counter()
            dets = raw_detect(args.detector, img, queries)
            entry["passes"][name] = {"queries": queries, "dets": dets, "seconds": time.perf_counter() - t0}
        cache[p] = entry
        if (i + 1) % 10 == 0:
            save_json(cache, out_path)            # resumable
        prog.update(i + 1)
    save_json(cache, out_path)
    print(f"Wrote {out_path} ({len(cache)} photos)")


if __name__ == "__main__":
    main()
