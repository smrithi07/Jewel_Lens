"""
Phase 7 — contrastive projection head on frozen DINOv2 embeddings (free labels: listing identity).

Data: artifacts/contrastive/views_train.npz — TRAIN-split images only, several views each
(clean + augmented, Phase 1 augmentations incl. 'combined'). Val is used only for early
stopping / choosing the config; test is touched once at the end (src/train/eval_head.py).

Pairs: for a listing, anchor = a random view of a random image; positive = a random view of a
DIFFERENT image of the same listing (p = 0.7, when it has >= 2 images) else another view of the
same image. Negatives: the other listings in the batch (symmetric InfoNCE, temperature tau).
Hard negatives (optional): each batch is drawn from ONE category, filled subtype by subtype,
so negatives look alike.

Heads: 'linear' = W x (W initialised to the identity), 'mlp' = x + MLP(x) (last layer zero-init),
both L2-normalised -> training starts exactly at the frozen baseline.

Selection on val: primary = mean of MRR(clean, all val queries) and MRR(combined phone-photo
augmentation, 300 subset), exact search over the projected catalogue, query image excluded.

Usage: python src/train/train_head.py            (grid: head x tau x hard negatives, ~10-20 min)
  -> artifacts/dinov2_ft/{head.pt, config.json}, results/phase7_train.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import copy
import itertools
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")
import config
from src.eval.common import save_json, load_json, SPLITS_PATH
from src.eval.retrieval_eval import Ctx, score, aggregate

VIEWS = os.environ.get("JL_VIEWS", "artifacts/contrastive/views_train.npz")
OUT = "artifacts/dinov2_ft"
BACKFILLED = "data/catalogue_metadata_backfilled.parquet"
SEED = 42
D = 768


class Head(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.lin = nn.Linear(D, D)
            nn.init.eye_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        else:
            self.mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        y = self.lin(x) if self.kind == "linear" else x + self.mlp(x)
        return F.normalize(y, dim=-1)


def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------ data

class Views:
    def __init__(self, emb_meta, listings, subtype):
        z = np.load(VIEWS)
        self.X = torch.from_numpy(l2n(z["X"].astype(np.float32)))
        fid = z["fid"]
        cid = emb_meta["catalogue_id"].to_numpy()
        self.listing_ids = sorted(set(cid[fid]))
        lpos = {c: i for i, c in enumerate(self.listing_ids)}
        self.views_by_img = defaultdict(list)
        for v, f in enumerate(fid):
            self.views_by_img[int(f)].append(v)
        self.imgs_by_listing = defaultdict(list)
        for f in sorted(self.views_by_img):
            self.imgs_by_listing[lpos[cid[f]]].append(f)
        self.cat = np.array([listings[c]["category"] for c in self.listing_ids])
        self.sub = np.array([subtype.get(c, "") or "_" for c in self.listing_ids])

    def batch(self, rng, P, hard, p_cross=0.7):
        n = len(self.listing_ids)
        if not hard:
            ls = rng.choice(n, size=P, replace=False)
        else:
            cats, counts = np.unique(self.cat, return_counts=True)
            c = rng.choice(cats, p=counts / counts.sum())
            pool = np.where(self.cat == c)[0]
            subs = rng.permutation(np.unique(self.sub[pool]))
            ls = []
            for s in subs:                                   # fill subtype by subtype -> look-alike negatives
                cand = rng.permutation(pool[self.sub[pool] == s])
                ls += list(cand[: P - len(ls)])
                if len(ls) >= P:
                    break
            ls = np.array(ls)
        a_idx, p_idx = [], []
        for l in ls:
            imgs = self.imgs_by_listing[l]
            ia = imgs[rng.integers(len(imgs))]
            va = self.views_by_img[ia][rng.integers(len(self.views_by_img[ia]))]
            if len(imgs) >= 2 and rng.random() < p_cross:
                ib = imgs[rng.integers(len(imgs))]
                while ib == ia:
                    ib = imgs[rng.integers(len(imgs))]
                vp = self.views_by_img[ib][rng.integers(len(self.views_by_img[ib]))]
            else:
                others = [v for v in self.views_by_img[ia] if v != va] or [va]
                vp = others[rng.integers(len(others))]
            a_idx.append(va)
            p_idx.append(vp)
        return self.X[a_idx], self.X[p_idx]


# ------------------------------------------------------------------ val evaluation (exact search)

class ValEval:
    def __init__(self, split="val"):
        from src.eval.embedding_eval import Index, query_sets
        self.ctx = Ctx(split, BACKFILLED)
        self.index = Index(self.ctx)
        self.base = torch.from_numpy(self.index.mats["dino"])
        qs = query_sets(self.ctx, self.index)
        self.sets = {a: (torch.from_numpy(np.stack([f["dino"] for _, f in v])), [q for q, _ in v])
                     for a, v in qs.items() if not a.startswith("_")}

    @torch.no_grad()
    def run(self, head, augs=("clean", "combined"), full=False):
        cat = head(self.base).numpy() if head is not None else self.base.numpy()
        out = {}
        for aug in augs:
            Q, qs = self.sets[aug]
            Qp = head(Q).numpy() if head is not None else Q.numpy()
            S = Qp @ cat.T
            rows = []
            for q, s in zip(qs, S):
                ranked = self.index.rank(s, q["fid"], "max")
                rows.append((q, score(self.ctx, q, ranked)))
            out[aug] = aggregate(rows, with_ci=full and aug == "clean") if full else \
                {"mrr": float(np.mean([m["mrr"] for _, m in rows])), "r1": float(np.mean([m["r1"] for _, m in rows]))}
        return out


def primary(r):
    get = lambda a: r[a]["all"]["mrr"] if "all" in r[a] else r[a]["mrr"]
    return 0.5 * (get("clean") + get("combined"))


# ------------------------------------------------------------------ training

def train_one(views, val, kind, tau, hard, P=256, max_steps=3000, eval_every=100, patience=6, lr=None, p_cross=0.7):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    head = Head(kind)
    opt = torch.optim.AdamW(head.parameters(), lr=lr or (3e-4 if kind == "linear" else 1e-3), weight_decay=1e-4)
    best = (primary(val.run(None)), 0, copy.deepcopy(head.state_dict()))   # step 0 = frozen baseline
    history, bad, t0 = [(0, best[0])], 0, time.time()
    for step in range(1, max_steps + 1):
        a, p = views.batch(rng, P, hard, p_cross)
        za, zp = head(a), head(p)
        logits = za @ zp.T / tau
        y = torch.arange(len(za))
        loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            head.eval()
            pr = primary(val.run(head))
            head.train()
            history.append((step, pr, loss.item()))
            if pr > best[0] + 1e-4:
                best, bad = (pr, step, copy.deepcopy(head.state_dict())), 0
            else:
                bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best[2])
    head.eval()
    return head, {"kind": kind, "tau": tau, "hard_negatives": hard, "best_val_primary": best[0], "best_step": best[1],
                  "history": history, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="linear,mlp")
    ap.add_argument("--taus", default="0.05,0.1")
    ap.add_argument("--positives", default="listing", choices=["listing", "same_image"],
                    help="listing: other images of the listing (p=0.7) or other views; same_image: only augmented views "
                         "of the same image (augmentation invariance, no shop-style shortcut)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    p_cross = 0.7 if args.positives == "listing" else 0.0
    torch.set_num_threads(1)
    L = load_json(SPLITS_PATH)["listings"]
    emb_meta = pd.read_parquet(config.CATALOGUE_EMBEDDINGS_META_PATH)
    meta = pd.read_parquet(BACKFILLED)
    subtype = dict(zip(meta["catalogue_id"], meta["subtype"].fillna("")))
    views = Views(emb_meta, L, subtype)
    val = ValEval("val")
    base = val.run(None)
    print(f"views {tuple(views.X.shape)} | train listings {len(views.listing_ids)} | frozen val primary {primary(base):.4f}", flush=True)

    runs, best = [], None
    for kind, tau, hard in itertools.product(args.heads.split(","), [float(t) for t in args.taus.split(",")], [False, True]):
        head, info = train_one(views, val, kind, tau, hard, p_cross=p_cross)
        runs.append(info)
        print(f"  {kind:6s} tau={tau:<5} hard={hard!s:5s} best val primary {info['best_val_primary']:.4f} "
              f"(step {info['best_step']}, {info['seconds']:.0f}s)", flush=True)
        if best is None or info["best_val_primary"] > best[1]["best_val_primary"]:
            best = (head, info)

    head, info = best
    OUTD = args.out
    os.makedirs(OUTD, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(OUTD, "head.pt"))
    save_json({**{k: v for k, v in info.items() if k != "history"}, "input": "l2_normalised_dinov2_cls",
               "trained_on": "train-split listings, views_train.npz", "seed": SEED, "positives": args.positives},
              os.path.join(OUTD, "config.json"))
    save_json({"frozen_val_primary": primary(base), "runs": runs, "chosen": {k: v for k, v in info.items() if k != "history"}},
              f"results/phase7_train_{os.path.basename(OUTD)}.json")
    print(f"chosen: {info['kind']} tau={info['tau']} hard={info['hard_negatives']} val primary "
          f"{primary(base):.4f} -> {info['best_val_primary']:.4f}")


if __name__ == "__main__":
    main()
