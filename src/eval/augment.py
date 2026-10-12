"""
Phase 1 — query-side augmentations that simulate phone photos of a
catalogue item. Applied to the QUERY image only, never to the index.
Every function is deterministic given the numpy Generator passed in.

Usage (preview grid): python src/eval/augment.py --n 6
  -> results/augment_preview.html
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF

AUG_MAX_SIDE = 1024   # queries are capped here before augmenting (DINOv2 sees 224px anyway)

AUGMENTATIONS = ["blur", "low_light", "bright_contrast", "rotate", "crop_zoom",
                 "jpeg", "busy_bg", "perspective"]
ALL_AUGS = ["clean"] + AUGMENTATIONS + ["combined"]

# Order used by "combined": geometry/scene first, then photometric, compression last.
_COMBINED_ORDER = ["crop_zoom", "busy_bg", "rotate", "perspective",
                   "low_light", "bright_contrast", "blur", "jpeg"]


def aug_rng(key, aug_name, seed=42):
    """Stable per-(query, augmentation) generator, independent of run order."""
    return np.random.default_rng([seed, zlib.crc32(f"{key}|{aug_name}".encode())])


def prepare(img):
    img = img.convert("RGB")
    if max(img.size) > AUG_MAX_SIDE:
        img = img.copy()
        img.thumbnail((AUG_MAX_SIDE, AUG_MAX_SIDE), Image.LANCZOS)
    return img


def border_color(img, frac=0.04):
    a = np.asarray(img)
    h, w = a.shape[:2]
    b = max(1, int(min(h, w) * frac))
    edge = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(edge, axis=0))


def _signed(rng, lo, hi):
    """Factor in [lo,hi] or its reciprocal range — i.e. strongly darker OR brighter."""
    f = rng.uniform(lo, hi)
    return f if rng.random() < 0.5 else 1.0 / f


def blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.006, 0.012) * max(img.size)))


def low_light(img, rng):
    a = np.asarray(img).astype(np.float32) / 255.0
    a = a ** rng.uniform(1.6, 2.4) * rng.uniform(0.35, 0.55)
    a = a * np.array([1.0, rng.uniform(0.92, 1.0), rng.uniform(0.75, 0.95)], dtype=np.float32)  # warm indoor cast
    a = a + rng.normal(0, rng.uniform(0.015, 0.035), a.shape).astype(np.float32)               # sensor noise
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def bright_contrast(img, rng):
    img = ImageEnhance.Brightness(img).enhance(_signed(rng, 0.55, 0.7))
    img = ImageEnhance.Contrast(img).enhance(_signed(rng, 0.55, 0.7))
    return ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))


def rotate(img, rng):
    angle = rng.uniform(10, 30) * (1 if rng.random() < 0.5 else -1)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=border_color(img))


def crop_zoom(img, rng):
    w, h = img.size
    area = rng.uniform(0.40, 0.75) * w * h
    ar = rng.uniform(3 / 4, 4 / 3)
    cw, ch = min(w, int(np.sqrt(area * ar))), min(h, int(np.sqrt(area / ar)))
    # crop centre stays in the middle half of the frame — phone shots are roughly centred
    cx = rng.uniform(0.25, 0.75) * w
    cy = rng.uniform(0.25, 0.75) * h
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)


def jpeg(img, rng):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(8, 21)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _busy_background(size, rng):
    w, h = size
    base = tuple(int(v) for v in rng.integers(40, 220, 3))
    bg = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(bg)
    for _ in range(int(rng.integers(40, 90))):
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        x0, y0 = rng.uniform(-0.1, 1.0) * w, rng.uniform(-0.1, 1.0) * h
        x1, y1 = x0 + rng.uniform(0.03, 0.4) * w, y0 + rng.uniform(0.03, 0.4) * h
        kind = rng.integers(3)
        if kind == 0:
            d.rectangle((x0, y0, x1, y1), fill=col)
        elif kind == 1:
            d.ellipse((x0, y0, x1, y1), fill=col)
        else:
            d.line((x0, y0, x1, y1), fill=col, width=int(rng.integers(2, max(3, w // 40))))
    a = np.asarray(bg).astype(np.float32)
    a += rng.normal(0, 12, a.shape)                          # texture
    bg = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return bg.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.5)))


def _foreground_mask(img):
    """Crude product mask: pixels that differ from the (usually plain) studio background."""
    a = np.asarray(img).astype(np.int16)
    diff = np.abs(a - np.array(border_color(img), dtype=np.int16)).max(axis=2)
    mask = Image.fromarray(((diff > 28) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
    cover = np.asarray(mask).mean() / 255
    if cover < 0.02 or cover > 0.9:      # no separable background (lifestyle/model shot)
        return Image.new("L", img.size, 255)
    return mask


def busy_bg(img, rng):
    w, h = img.size
    scale = rng.uniform(1.3, 1.8)
    bg = _busy_background((int(w * scale), int(h * scale)), rng)
    mask = _foreground_mask(img)
    x0 = int(rng.uniform(0, bg.width - w))
    y0 = int(rng.uniform(0, bg.height - h))
    bg.paste(img, (x0, y0), mask)
    return bg


def perspective(img, rng):
    w, h = img.size
    d = rng.uniform(0.15, 0.30)
    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[int(x + rng.uniform(-d, d) * w * 0.5 + (d * w * 0.25 if x == 0 else -d * w * 0.25)),
            int(y + rng.uniform(-d, d) * h * 0.5 + (d * h * 0.25 if y == 0 else -d * h * 0.25))]
           for x, y in start]
    return TF.perspective(img, start, end, interpolation=TF.InterpolationMode.BILINEAR,
                          fill=list(border_color(img)))


_FUNCS = {"blur": blur, "low_light": low_light, "bright_contrast": bright_contrast, "rotate": rotate,
          "crop_zoom": crop_zoom, "jpeg": jpeg, "busy_bg": busy_bg, "perspective": perspective}


def combined(img, rng):
    """3 distinct augmentations, applied in a realistic order. Photometric pair is exclusive."""
    pool = list(AUGMENTATIONS)
    picks = set()
    while len(picks) < 3:
        a = pool[int(rng.integers(len(pool)))]
        if {a} | picks >= {"low_light", "bright_contrast"}:
            continue
        picks.add(a)
    for name in _COMBINED_ORDER:
        if name in picks:
            img = _FUNCS[name](img, rng)
    return img


def apply_aug(name, img, key):
    """Prepared (capped) image -> augmented image. `key` makes it reproducible per query."""
    img = prepare(img)
    if name == "clean":
        return img
    rng = aug_rng(key, name)
    if name == "combined":
        return combined(img, rng)
    return _FUNCS[name](img, rng)


if __name__ == "__main__":
    import argparse, sys
    sys.path.append(".")
    from src.eval.common import load_json, thumb_b64, SPLITS_PATH, open_image
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    listings = load_json(SPLITS_PATH)["listings"]
    qs = [v["query_image"] for v in listings.values() if v["split"] == "val" and v["eligible"]][: args.n]
    rows = []
    for q in qs:
        img = open_image(q, AUG_MAX_SIDE)
        cells = "".join(f'<td><img src="{thumb_b64(apply_aug(a, img, q), 170)}"><br>{a}</td>' for a in ALL_AUGS)
        rows.append(f"<tr>{cells}</tr>")
    os.makedirs("results", exist_ok=True)
    with open("results/augment_preview.html", "w") as f:
        f.write("<html><body style='font-family:sans-serif;font-size:11px'><table>" + "".join(rows) + "</table></body></html>")
    print("Wrote results/augment_preview.html")
