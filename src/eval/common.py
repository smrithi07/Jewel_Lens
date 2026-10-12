"""
Shared helpers for the v2 evaluation scripts: paths, JSON-safe casting,
bootstrap confidence intervals, base64 thumbnails, progress/ETA printing
and appending markdown sections to results/LOG.md.
"""

import base64
import json
import time
from io import BytesIO

import numpy as np
from PIL import Image

SPLITS_PATH = "data/splits.json"
STUMPER_SPLIT_PATH = "data/stumper/split.json"
STUMPER_ROOT = "data/stumper/images"
EXTRA_NEGATIVES_DIR = "data/negatives"
PHASH_CACHE_PATH = "data/phash_cache.parquet"
RESULTS_DIR = "results"
LOG_PATH = "results/LOG.md"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def to_jsonable(obj):
    """Recursively cast numpy scalars/arrays so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_boot=1000, seed=42, alpha=0.05):
    """Mean and percentile CI of a per-unit metric (e.g. per-query or per-photo)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(len(values)),
    }


def fmt_ci(ci, digits=3):
    if ci is None or ci.get("mean") is None:
        return "–"
    return f"{ci['mean']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def open_image(path, max_side=None):
    """Open as RGB; uses JPEG draft mode for fast decode when downscaling."""
    img = Image.open(path)
    if max_side and img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def thumb_b64(img_or_path, max_size=130):
    try:
        img = open_image(img_or_path, max_size * 2) if isinstance(img_or_path, str) else img_or_path.copy()
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


class Progress:
    """Prints 'label i/n (pct) | rate | ETA' at most every `every` seconds."""

    def __init__(self, total, label, every=5.0):
        self.total, self.label, self.every = total, label, every
        self.start = self.last = time.time()

    def update(self, done):
        now = time.time()
        if now - self.last < self.every and done < self.total:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else 0
        print(f"  [{self.label}] {done}/{self.total} ({100 * done / max(self.total, 1):.0f}%) "
              f"| {rate:.1f}/s | ETA {eta / 60:.1f} min", flush=True)


def append_log(markdown):
    with open(LOG_PATH, "a") as f:
        f.write("\n" + markdown.rstrip() + "\n")


def replace_log_section(header_prefix, markdown):
    """Replace the LOG.md section whose '## ' header starts with header_prefix (or append if absent)."""
    try:
        text = open(LOG_PATH).read()
    except FileNotFoundError:
        text = ""
    start = text.find("\n" + header_prefix)
    if start == -1:
        return append_log(markdown)
    nxt = text.find("\n## ", start + 1)
    end = nxt if nxt != -1 else len(text)
    with open(LOG_PATH, "w") as f:
        f.write(text[:start] + "\n" + markdown.rstrip() + "\n" + text[end:])
