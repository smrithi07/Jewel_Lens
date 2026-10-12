"""
Phase 7 — evaluate the trained projection head against the frozen baseline, and build its index.

  1. artifacts/dinov2_ft/: projected catalogue embeddings + FAISS IndexHNSWFlat (same params
     as production: M=32, efC=200, efS=64, IP, IDMap; same faiss ids) + id mapping copy
  2. val: every augmentation, without / with the Phase 4 fused re-rank, frozen A vs FT
  3. own photos (stumper, tune + holdout): the production boxes (Phase 4 baseline detections),
     crop -> DINOv2 -> (head) -> exact search -> fused category (kNN over the SAME index's
     neighbours) -> fused re-rank -> top-5. Category accuracy + graded retrieval; results
     never graded are listed for a top-up grading round
  4. test (--test, once): same as 2 on the test split; refuses to run twice

Usage: python src/train/eval_head.py           (val + own photos + index)
       python src/train/eval_head.py --test    (the one test run, after the user has seen val)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import pickle
import shutil
import sys
import time
from datetime import datetime

import faiss
import numpy as np
import torch

sys.path.append(".")
import config
from src.eval.common import save_json, load_json, fmt_ci, append_log, STUMPER_SPLIT_PATH
from src.eval.augment import ALL_AUGS
from src.eval.embedding_eval import Index, query_sets, evaluate, primary, BACKFILLED
from src.eval.retrieval_eval import Ctx
from src.eval import own_eval
from src.train.train_head import Head
from src.pipeline.retrieve import apply_rerank
from src.pipeline.decision import decide
from src.pipeline.category import categorize, load_category_models

FT = os.environ.get("JL_FT_DIR", "artifacts/dinov2_ft")
TAG = os.path.basename(FT).replace("dinov2_", "")          # "ft" / "ftaug" -> output file names


def load_head():
    cfg = load_json(os.path.join(FT, "config.json"))
    head = Head(cfg["kind"])
    head.load_state_dict(torch.load(os.path.join(FT, "head.pt")))
    head.eval()
    return head, cfg


@torch.no_grad()
def project(head, X):
    X = np.asarray(X, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    return head(torch.from_numpy(X)).numpy()


def build_index(head):
    raw = np.load(config.CATALOGUE_EMBEDDINGS_PATH)
    P = project(head, raw).astype(np.float32)
    np.save(os.path.join(FT, "embeddings_proj.npy"), P)
    base = faiss.IndexHNSWFlat(P.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = 200
    base.hnsw.efSearch = 64
    idx = faiss.IndexIDMap(base)
    idx.add_with_ids(P, np.arange(len(P)).astype("int64"))
    faiss.write_index(idx, os.path.join(FT, "catalogue.faiss"))
    shutil.copy(config.ID_MAPPING_PATH, os.path.join(FT, "id_mapping.pkl"))
    return P


def catalogue_eval(split, head):
    ctx = Ctx(split, BACKFILLED)
    index = Index(ctx)
    index.mats["ft"] = project(head, index.mats["dino"])
    qs = query_sets(ctx, index)
    for aug, items in qs.items():
        if aug.startswith("_"):
            continue
        D = np.stack([f["dino"] for _, f in items])
        Pq = project(head, D)
        for (q, f), p in zip(items, Pq):
            f["ft"] = p
    probs = {}
    out = {}
    for name, spec in (("A frozen", {"dino": 1.0}), ("FT", {"ft": 1.0})):
        r = evaluate(index, ctx, qs, spec, "max", probs)
        out[name] = {"res": r, "primary": primary(r), "primary_fused": primary(r, "fused")}
    return out


def own_photo_eval(head, halves=("tune", "holdout")):
    """Production boxes; retrieval + fused category from each variant's own neighbours.
    Each result carries the listing image that actually matched (shown on grading pages)."""
    from src.eval.detection_bench import CropCache
    load_category_models()
    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    dets = load_json("data/eval_own/phase4_live_detections.json")
    ctx = Ctx("val", BACKFILLED)
    index = Index(ctx)
    index.mats["ft"] = project(head, index.mats["dino"])
    crops = CropCache()
    labels, graded = own_eval.load_labels(), own_eval.load_grades()
    out, files, lat = {}, {}, []
    pos = {c: i for i, c in enumerate(index.cid)}
    paths = ctx.emb_meta["image_path"].to_numpy()
    for name, key in (("A frozen", "dino"), ("FT", "ft")):
        dd = {}
        for p, d in dets.items():
            items = []
            for it in d["items"]:
                f = crops.get(p, it["box"])
                raw = f["raw"]
                norm = raw / np.linalg.norm(raw)
                t = time.perf_counter()
                q = project(head, raw[None])[0] if key == "ft" else norm
                if key == "ft":
                    lat.append(time.perf_counter() - t)
                sims = index.mats[key] @ q
                ranked = index.rank(sims, None, "max")
                for r in ranked:
                    li = pos[r["catalogue_id"]]
                    a, n = index.starts[li], index.counts[li]
                    r["image_path"] = paths[a + int(np.argmax(sims[a:a + n]))]
                fused = categorize(None, norm, raw, it["detector_category"], ranked, clip_emb=f["clip"])
                rr = apply_rerank([dict(r) for r in ranked], fused["probs"])
                items.append({**{k: it[k] for k in ("item_index", "box", "detector_category", "detector_score")},
                              "classifier_category": fused["category"], "classifier_confidence": fused["confidence"],
                              "top5": rr[:5], "status": decide(rr)["status"]})
            dd[p] = {**{k: d[k] for k in ("folder", "half", "size")}, "items": items}
        path = f"data/eval_own/phase7_{TAG if key == 'ft' else key}_detections.json"
        save_json(dd, path)
        files[name] = path
        out[name] = {h: own_eval.evaluate(dd, split, labels, graded, h, "classifier_category") for h in halves}
    crops.save()
    return out, files, float(np.median(lat)) if lat else None


def md_catalogue(res, split):
    L = [f"**Catalogue {split} (exact search, same-listing protocol)**", "",
         "| variant | primary | primary (fused) | R@1 | R@5 | R@10 | MRR | nDCG@5 | graded nDCG@5 (trusted) | cat P@5 (non-set) | colour P@5 (trusted) | subtype P@5 (trusted) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        c = r["res"]["clean"]["none"]["all"]
        L.append(f"| {n} | {r['primary']:.4f} | {r['primary_fused']:.4f} | {fmt_ci(c['ci']['r1'])} | {c['r5']:.3f} | {c['r10']:.3f} | "
                 f"{fmt_ci(c['ci']['mrr'])} | {c['ndcg5']:.3f} | {c.get('graded_ndcg5_trusted', 0):.3f} | {c['cat_p5_non_set']:.3f} | "
                 f"{c.get('color_p5_trusted', 0):.3f} | {c.get('subtype_p5_trusted', 0):.3f} |")
    L += ["", f"Per augmentation ({split}, R@1 / MRR, no re-rank):", "", "| aug | A frozen | FT |", "|---|---|---|"]
    for aug in ALL_AUGS:
        a, f = res["A frozen"]["res"].get(aug), res["FT"]["res"].get(aug)
        if a and f:
            L.append(f"| {aug} | {a['none']['all']['r1']:.3f} / {a['none']['all']['mrr']:.3f} | "
                     f"{f['none']['all']['r1']:.3f} / {f['none']['all']['mrr']:.3f} |")
    L += ["", "Per category (clean, non-set, R@1 / MRR):", "", "| category | A frozen | FT |", "|---|---|---|"]
    for c in config.CATEGORIES:
        a, f = res["A frozen"]["res"]["clean"]["none"]["by_category"][c], res["FT"]["res"]["clean"]["none"]["by_category"][c]
        L.append(f"| {c} | {a['r1']:.3f} / {a['mrr']:.3f} | {f['r1']:.3f} / {f['mrr']:.3f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--halves", default="tune,holdout", help="own-photo halves to score (tune only while choosing)")
    args = ap.parse_args()
    halves = tuple(args.halves.split(","))
    head, cfg = load_head()
    if args.test:
        path = f"results/phase7_{TAG}_test.json" if TAG != "ft" else "results/phase7_test.json"
        if os.path.exists(path):
            sys.exit(f"{path} exists — the test split is evaluated once.")
        res = catalogue_eval("test", head)
        save_json({"created": datetime.now().isoformat(timespec="seconds"), "head": cfg, "test": res}, path)
        md = f"## Phase 7 — test split, evaluated once ({datetime.now().isoformat(timespec='seconds')})\n\n" + md_catalogue(res, "test")
        append_log(md)
        print(md)
        return

    P = build_index(head)
    val = catalogue_eval("val", head)
    own, files, proj_lat = own_photo_eval(head, halves)
    rep = {"created": datetime.now().isoformat(timespec="seconds"), "head": cfg, "val": val, "own": own,
           "own_files": files, "projection_latency_ms": (proj_lat or 0) * 1000,
           "index": {"path": f"{FT}/catalogue.faiss", "dim": int(P.shape[1]), "mb": P.nbytes / 1e6}}
    save_json(rep, "results/phase7.json" if TAG == "ft" else f"results/phase7_{TAG}.json")
    L = [f"## Phase 7 — contrastive projection head [{TAG}] ({rep['created']})", "",
         f"Head: {cfg['kind']}, tau={cfg['tau']}, hard negatives={cfg['hard_negatives']}, best step {cfg['best_step']} "
         f"(chosen on val). Trained on train-split listings only. Projection adds {rep['projection_latency_ms']:.2f} ms per query; "
         f"index {rep['index']['mb']:.0f} MB, dim {rep['index']['dim']}.", "", md_catalogue(val, "val"), "",
         "**Own photos (production boxes; retrieval and fused category from each variant's own neighbours)**", "",
         "| metric | half | A frozen | FT |", "|---|---|---|---|"]
    for h in halves:
        a, f = own["A frozen"][h], own["FT"][h]
        L.append(f"| Category acc (first item) | {h} | {fmt_ci(a['category_acc_first'])} | {fmt_ci(f['category_acc_first'])} |")
        L.append(f"| Item-category F1 | {h} | {fmt_ci(a['item_category_f1'])} | {fmt_ci(f['item_category_f1'])} |")
    for k, n in (("ndcg5", "Graded nDCG@5"), ("p5", "P@5"), ("s1", "Success@1"), ("judged", "judged@5")):
        for h in halves:
            if own["A frozen"][h]["graded"]["n_items"]:
                L.append(f"| {n} | {h} | {fmt_ci(own['A frozen'][h]['graded'][k])} | {fmt_ci(own['FT'][h]['graded'][k])} |")
    append_log("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
