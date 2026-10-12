"""
Phase 8 — open-set calibration of the match / uncertain / no_match decision.

Signals per detected item (production path: Phase 4 fused category + fused re-rank, v1 boxes)
  top1   : post-re-rank top-1 score (what src/pipeline/decision.py sees)
  margin : top1 - top2
  p_oos  : CLIP out-of-scope probability of the crop: softmax(100 cos) over the 3 in-scope
           category prompts (artifacts/category_v2/clip_text.npy) + a FIXED list of out-of-scope
           prompts (rings, watches, anklets, brooches, nose pins, hair accessories, everyday
           objects), summed over the out-of-scope ones. The prompt list is written down a priori,
           never tuned on negatives. Live cost: one matrix product on the CLIP crop embedding the
           fused category already computes.

Negatives are too few to fit on (tune: 8 photos), so thresholds control the FALSE-REJECT rate
on the plentiful tune POSITIVE photos (Neyman-Pearson style), and the false-accept rate is
reported on held-out negatives:
  T_low : a positive photo is falsely rejected when every item has top1 < T_low; chosen so that
          tune FRR = FRR_BUDGET / 2
  tau   : ... when every item has p_oos > tau; chosen so that tune FRR = FRR_BUDGET / 2
  T_high (the "closest matches" tier): smallest score with P(grade >= 2) >= 0.9 over all graded
          results on your TUNE photos; M (top-1 margin for "match" vs "uncertain") adopted only if it
          adds >= 0.05 precision at >= half the coverage. Holdout is a check only.

Usage: python src/eval/calibrate.py   -> results/phase8_calibration.json, LOG.md, artifacts/openset/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
from datetime import datetime

import numpy as np

sys.path.append(".")
import config
from src.eval.common import STUMPER_SPLIT_PATH, bootstrap_ci, fmt_ci, load_json, save_json, replace_log_section
from src.eval import own_eval
from src.eval.detection_bench import CropCache
from src.catalogue.build_clip_embeddings import clip_text_embed

FRR_BUDGET = 0.05
PREC_TARGET = 0.9
OUT = "artifacts/openset"
DETS = "data/eval_own/phase4_live_detections.json"
OOS_PROMPTS = ["a ring", "a finger ring", "a ring worn on a finger", "a wristwatch", "a watch", "an anklet",
               "an anklet worn on the ankle", "a brooch", "a nose pin", "a nose ring", "a hair clip", "a hair band",
               "a scrunchie", "a keychain", "a pair of glasses", "a bottle", "a shoe", "a toy", "a cup",
               "a piece of fabric", "an everyday object"]
TEMPLATES = ["a photo of {}.", "a close-up photo of {}."]


def oos_text():
    rows = []
    for p in OOS_PROMPTS:
        e = clip_text_embed([t.format(p) for t in TEMPLATES]).mean(0)
        rows.append(e / np.linalg.norm(e))
    return np.stack(rows)


def p_oos(clip_emb, in_text, out_text):
    z = 100.0 * np.concatenate([in_text @ clip_emb, out_text @ clip_emb])
    z = np.exp(z - z.max())
    p = z / z.sum()
    return float(p[len(in_text):].sum())


def photo_records(dets, split, crops, in_text, out_text):
    recs = []
    for p, d in dets.items():
        items = []
        for it in d["items"]:
            s = [r["score"] for r in it["top5"]]
            items.append({"box": it["box"], "top1": s[0] if s else 0.0, "margin": (s[0] - s[1]) if len(s) > 1 else 0.0,
                          "top1_id": it["top5"][0]["catalogue_id"] if s else None,
                          "p_oos": p_oos(crops.get(p, it["box"])["clip"], in_text, out_text)})
        recs.append({"photo": p, "folder": d["folder"], "half": split[p]["half"], "items": items,
                     "negative_type": own_eval.negative_type_guess(p) if d["folder"] == "negatives" else None})
    return recs


def rejected(rec, t_low, tau):
    """A photo is rejected when every item is rejected (no item reaches the user)."""
    return all(it["top1"] < t_low or it["p_oos"] > tau for it in rec["items"])


def threshold_for_frr(values, target):
    """values: per positive photo, the best item's score (top1) — reject below t. Quantile at target."""
    return float(np.quantile(np.asarray(values), target))


def main():
    split = load_json(STUMPER_SPLIT_PATH)["photos"]
    dets = load_json(DETS)
    crops = CropCache()
    in_text = np.load("artifacts/category_v2/clip_text.npy")
    out_text = oos_text()
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "oos_text.npy"), out_text)
    recs = photo_records(dets, split, crops, in_text, out_text)
    crops.save()

    pos = lambda h: [r for r in recs if r["half"] == h and r["folder"] != "negatives" and r["items"]]
    neg = lambda h: [r for r in recs if r["half"] == h and r["folder"] == "negatives"]

    # --- T_low and tau from tune positives (each gets half the FRR budget)
    best_top1 = [max(it["top1"] for it in r["items"]) for r in pos("tune")]
    min_oos = [min(it["p_oos"] for it in r["items"]) for r in pos("tune")]
    t_low = threshold_for_frr(best_top1, FRR_BUDGET / 2)
    tau = float(np.quantile(min_oos, 1 - FRR_BUDGET / 2))

    def rates(h, tl, ta):
        P, N = pos(h), neg(h)
        frr = [float(rejected(r, tl, ta)) for r in P]
        far = [float(not rejected(r, tl, ta)) if r["items"] else 0.0 for r in N]
        by = {t: bootstrap_ci([float(not rejected(r, tl, ta)) if r["items"] else 0.0 for r in N if r["negative_type"] == t])
              for t in sorted({r["negative_type"] for r in N})}
        return {"frr": bootstrap_ci(frr), "far": bootstrap_ci(far), "far_by_type": by, "n_pos": len(P), "n_neg": len(N)}

    old = {"t_low": config.LOW_MATCH_THRESHOLD, "tau": 1.01}
    configs = {"v1 (LOW=0.45, no OOS check)": old, "similarity only (calibrated T_low)": {"t_low": t_low, "tau": 1.01},
               "OOS check only (calibrated tau)": {"t_low": -1, "tau": tau}, "calibrated T_low + OOS check": {"t_low": t_low, "tau": tau}}
    table = {n: {h: rates(h, c["t_low"], c["tau"]) for h in ("tune", "holdout")} for n, c in configs.items()}

    # FAR/FRR curve over T_low (similarity only and with the OOS check), tune + holdout
    curve = []
    for t in np.round(np.arange(0.30, 0.86, 0.02), 2):
        curve.append({"t_low": float(t), **{f"{h}_{k}": rates(h, t, ta)[k]["mean"] for h in ("tune", "holdout")
                                             for k in ("frr", "far") for ta in [tau]},
                      **{f"{h}_far_simonly": rates(h, t, 1.01)["far"]["mean"] for h in ("tune", "holdout")}})

    # --- closest-match tier. Per graded RESULT (all ranks, more data than top-1 only): P(grade >= 2,
    # "same design family or better") as a function of its score. Grade 3 is not used: its use drifted
    # between grading sessions (23% of tune pooled judgements vs 83% of holdout Phase-3 judgements).
    graded = own_eval.load_grades()

    def result_pairs(half):
        res, top1 = [], []
        for r in pos(half):
            for it_rec, it in zip(r["items"], dets[r["photo"]]["items"]):
                g = next((x for x in graded if x["photo"] == r["photo"] and not x["bad_crop"]
                          and own_eval.iou(x["box"], it["box"]) >= 0.5), None)
                if not g:
                    continue
                for k, rr in enumerate(it["top5"][:5]):
                    if rr["catalogue_id"] in g["grades"]:
                        ok = g["grades"][rr["catalogue_id"]] >= 2
                        res.append((rr["score"], ok))
                        if k == 0:
                            top1.append((rr["score"], it_rec["margin"], ok))
        return np.array(res, float), np.array(top1, float)

    tune_res, tune_top1 = result_pairs("tune")
    hold_res, hold_top1 = result_pairs("holdout")

    def prec_curve(a, min_n=10):
        out = {}
        for T in np.round(np.arange(0.45, 0.9, 0.01), 2):
            m = a[:, 0] >= T
            if m.sum() >= min_n:
                out[float(T)] = (float(a[m, 1].mean()), int(m.sum()))
        return out

    tc, hc = prec_curve(tune_res), prec_curve(hold_res, min_n=1)
    ok = sorted(T for T, (p_, n_) in tc.items() if p_ >= PREC_TARGET)
    T_high = ok[0] if ok else max(tc)
    # margin on the top-1 decides "match" vs "uncertain": adopt only if it adds >= 0.05 precision at >= half coverage
    M_best = 0.0
    base = tune_top1[tune_top1[:, 0] >= T_high]
    p0 = float(base[:, 2].mean()) if len(base) else 0.0
    for M in (0.01, 0.02, 0.03, 0.05):
        m = (tune_top1[:, 0] >= T_high) & (tune_top1[:, 1] >= M)
        if m.sum() >= max(5, 0.5 * len(base)) and tune_top1[m, 2].mean() >= p0 + 0.05:
            M_best, p0 = M, float(tune_top1[m, 2].mean())
    closest = {"T_high": float(T_high), "margin": float(M_best), "criterion": "P(grade >= 2) >= target, per graded result",
               "target": PREC_TARGET, "tune_curve": tc, "holdout_check": hc.get(float(T_high)),
               "holdout_curve": hc, "n_tune_results": len(tune_res), "n_holdout_results": len(hold_res),
               "v1_0.65": {"tune": tc.get(0.65), "holdout": hc.get(0.65)}}

    calib = {"LOW_MATCH_THRESHOLD": round(t_low, 3), "OOS_THRESHOLD": round(tau, 3), "HIGH_MATCH_THRESHOLD": round(T_high, 3),
             "MARGIN_THRESHOLD": round(M_best, 3), "FRR_BUDGET": FRR_BUDGET}
    rep = {"created": datetime.now().isoformat(timespec="seconds"), "calibrated": calib, "table": table, "curve": curve,
           "closest_tier": closest, "oos_prompts": OOS_PROMPTS,
           "records": [{**r, "items": [{k: v for k, v in it.items() if k != "box"} for it in r["items"]]} for r in recs]}
    save_json(rep, "results/phase8_calibration.json")
    save_json(calib, os.path.join(OUT, "calibration.json"))
    md = markdown(rep)
    replace_log_section("## Phase 8 — open-set calibration", md)
    print(md)


def val_top1_pairs():
    """Catalogue val (exact frozen search + fused re-rank): (top1, margin, top-1 is same listing or pHash near-dup)."""
    from src.eval.embedding_eval import Index, BACKFILLED, fused_probs
    from src.eval.retrieval_eval import Ctx
    from src.pipeline.retrieve import apply_rerank
    from src.pipeline.category import load_category_models
    load_category_models()
    ctx = Ctx("val", BACKFILLED)
    index = Index(ctx)
    raw = np.load(config.CATALOGUE_EMBEDDINGS_PATH, mmap_mode="r")
    pairs = []
    for q in ctx.queries:
        f = {"dino": index.mats["dino"][q["fid"]], "clip": index.mats["clip"][q["fid"]]}
        ranked = index.rank(index.mats["dino"] @ f["dino"], q["fid"], "max")
        rr = apply_rerank([dict(r) for r in ranked], fused_probs(index, ctx, q, f, np.asarray(raw[q["fid"]], np.float32)))
        rel = ctx.near_dup_listings(q["fid"]) | {q["cid"]}
        pairs.append((rr[0]["score"], rr[0]["score"] - rr[1]["score"], rr[0]["catalogue_id"] in rel))
    return pairs


def markdown(rep):
    c = rep["calibrated"]
    L = [f"## Phase 8 — open-set calibration ({rep['created']})", "",
         f"Thresholds fixed by a **false-reject budget of {c['FRR_BUDGET']:.0%} on tune positive photos** (half each for similarity "
         "and the CLIP out-of-scope check); false-accept rate reported on negatives (holdout has 9 negatives, so CIs are wide). "
         "Out-of-scope prompts were written down a priori, not tuned.", "",
         f"Calibrated: LOW_MATCH_THRESHOLD **{c['LOW_MATCH_THRESHOLD']}**, OOS_THRESHOLD **{c['OOS_THRESHOLD']}**, "
         f"HIGH_MATCH_THRESHOLD **{c['HIGH_MATCH_THRESHOLD']}**, MARGIN_THRESHOLD **{c['MARGIN_THRESHOLD']}** "
         "(v1: 0.45 / none / 0.65 / 0.03).", "",
         "| decision rule | half | false-reject rate (positive photos) | false-accept rate (negative photos) | FAR non-jewellery | FAR out-of-scope jewellery |",
         "|---|---|---|---|---|---|"]
    for n, v in rep["table"].items():
        for h in ("tune", "holdout"):
            x = v[h]
            L.append(f"| {n} | {h} | {fmt_ci(x['frr'])} (n={x['n_pos']}) | {fmt_ci(x['far'])} (n={x['n_neg']}) | "
                     f"{fmt_ci(x['far_by_type'].get('non_jewellery'))} | {fmt_ci(x['far_by_type'].get('out_of_scope_jewellery'))} |")
    ct = rep["closest_tier"]
    L += ["", f"Closest-matches tier: result score ≥ **{ct['T_high']}** (margin rule: {ct['margin']}). Criterion: P(grade ≥ 2) ≥ "
          f"{ct['target']} over every graded result on tune photos (n={ct['n_tune_results']}). Tune at threshold: "
          f"{ct['tune_curve'].get(ct['T_high'])}; holdout check: {ct['holdout_check']} (precision, n). v1 threshold 0.65: {ct['v1_0.65']}.", "",
          "FAR/FRR curve (T_low sweep; with the calibrated OOS check):", "",
          "| T_low | tune FRR | tune FAR | holdout FRR | holdout FAR | holdout FAR (similarity only) |", "|---|---|---|---|---|---|"]
    for r in rep["curve"][::2]:
        L.append(f"| {r['t_low']} | {r['tune_frr']:.3f} | {r['tune_far']:.3f} | {r['holdout_frr']:.3f} | {r['holdout_far']:.3f} | {r['holdout_far_simonly']:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
