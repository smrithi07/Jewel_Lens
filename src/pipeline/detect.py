"""
Phase 3 — OWL-ViT zero-shot jewelry detection.
Given an image, returns a list of {category, box, score} for each
detected item above the confidence threshold. No training needed.
"""

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import sys
sys.path.append(".")
import config
from torchvision.ops import nms


_processor = None
_model = None


def load_detector():
    global _processor, _model
    if _model is None:
        print(f"Loading detector: {config.DETECTOR_MODEL} ...")
        _processor = OwlViTProcessor.from_pretrained(config.DETECTOR_MODEL)
        _model = OwlViTForObjectDetection.from_pretrained(config.DETECTOR_MODEL)
        _model.eval()
    return _processor, _model


def detect(image, prompts=None, threshold=None, nms_iou_threshold=0.3):
    processor, model = load_detector()
    prompts = prompts or config.DETECTION_PROMPTS
    threshold = threshold if threshold is not None else config.DETECTION_CONFIDENCE_THRESHOLD

    # use descriptive templates instead of bare "a photo of a {p}"
    text_prompts = [config.DETECTION_PROMPT_TEMPLATES.get(p, f"a photo of a {p}") for p in prompts]
    text_queries = [text_prompts]

    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    if len(boxes) == 0:
        return []

    keep_idx = nms(boxes, scores, iou_threshold=nms_iou_threshold)

    detections = []
    for i in keep_idx:
        detections.append({
            "category": prompts[labels[i]],  # still index into original prompts list for the category name
            "box": tuple(boxes[i].tolist()),
            "score": float(scores[i]),
        })
    return detections


def crop_with_padding(image, box, padding_ratio=None):
    padding_ratio = padding_ratio if padding_ratio is not None else config.CROP_PADDING_RATIO
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pad_w, pad_h = w * padding_ratio, h * padding_ratio
    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(image.width, x1 + pad_w)
    y1 = min(image.height, y1 + pad_h)
    return image.crop((x0, y0, x1, y1))