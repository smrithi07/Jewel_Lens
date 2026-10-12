"""
Phase 5 — raw zero-shot detection for three detectors with one interface.

raw_detect(name, image, queries) -> list of {"box": [x0,y0,x1,y1], "scores": [score per query]}
  - no thresholding beyond a low floor, no NMS: post-processing lives in
    src/pipeline/postprocess.py so every rule can be toggled and ablated
  - OWL-ViT / OWLv2: one forward pass scores every box against every query
  - Grounding DINO: one pass over a ". "-joined prompt; each box gets the score of the
    phrase it was grounded to (other queries 0)

Detectors: "owlvit" (google/owlvit-base-patch32, current), "owlv2"
(google/owlv2-base-patch16-ensemble), "gdino" (IDEA-Research/grounding-dino-tiny).
"""

import sys

import numpy as np
import torch

sys.path.append(".")

MODEL_IDS = {"owlvit": "google/owlvit-base-patch32",
             "owlv2": "google/owlv2-base-patch16-ensemble",
             "gdino": "IDEA-Research/grounding-dino-tiny"}
SCORE_FLOOR = 0.03
MAX_BOXES = 100
_cache = {}


def load(name):
    if name in _cache:
        return _cache[name]
    mid = MODEL_IDS[name]
    if name == "owlvit":
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        p, m = OwlViTProcessor.from_pretrained(mid), OwlViTForObjectDetection.from_pretrained(mid)
    elif name == "owlv2":
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        p, m = Owlv2Processor.from_pretrained(mid), Owlv2ForObjectDetection.from_pretrained(mid)
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        p, m = AutoProcessor.from_pretrained(mid), AutoModelForZeroShotObjectDetection.from_pretrained(mid)
    _cache[name] = (p, m.eval())
    return _cache[name]


def _cxcywh_to_xyxy(b, sx, sy):
    cx, cy, w, h = b.T
    return np.stack([(cx - w / 2) * sx, (cy - h / 2) * sy, (cx + w / 2) * sx, (cy + h / 2) * sy], axis=1)


def _clip(boxes, W, H):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    return boxes


def raw_detect(name, image, queries):
    p, m = load(name)
    W, H = image.size
    if name in ("owlvit", "owlv2"):
        inputs = p(text=[queries], images=image, return_tensors="pt")
        with torch.no_grad():
            out = m(**inputs)
        scores = torch.sigmoid(out.logits[0]).numpy()                     # (N boxes, Q)
        boxes = out.pred_boxes[0].numpy()
        # OWL-ViT resizes to a square (boxes relative to W,H); OWLv2 pads to a square first
        sx, sy = (W, H) if name == "owlvit" else (max(W, H), max(W, H))
        boxes = _clip(_cxcywh_to_xyxy(boxes, sx, sy), W, H)
        keep = np.argsort(-scores.max(1))[:MAX_BOXES]
        keep = keep[scores[keep].max(1) >= SCORE_FLOOR]
        return [{"box": boxes[i].round(1).tolist(), "scores": scores[i].round(4).tolist()} for i in keep]

    # Grounding DINO
    text = ". ".join(q.rstrip(".") for q in queries) + "."
    inputs = p(text=text, images=image, return_tensors="pt")
    with torch.no_grad():
        out = m(**inputs)
    res = p.post_process_grounded_object_detection(out, inputs.input_ids, threshold=SCORE_FLOOR,
                                                   text_threshold=SCORE_FLOOR, target_sizes=[(H, W)])[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = []
    for box, score, lab in zip(res["boxes"].numpy(), res["scores"].numpy(), labels):
        s = [0.0] * len(queries)
        lab = str(lab).strip().lower()
        # the grounded phrase is a (sub)span of one query: pick the query sharing most words
        best = max(range(len(queries)), key=lambda i: len(set(lab.split()) & set(queries[i].lower().split())))
        s[best] = float(score)
        dets.append({"box": np.array(box).clip([0, 0, 0, 0], [W, H, W, H]).round(1).tolist(), "scores": s})
    dets.sort(key=lambda d: -max(d["scores"]))
    return dets[:MAX_BOXES]
