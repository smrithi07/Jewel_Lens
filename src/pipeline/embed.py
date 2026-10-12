"""
Phase 4a — DINOv2 embedding for a single query crop.
Reuses the same model/preprocessing as the catalogue embedding step,
so query and catalogue vectors live in the same space.
"""

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.append(".")
import config

_processor = None
_model = None


def load_embedder():
    global _processor, _model
    if _model is None:
        print(f"Loading embedder: {config.EMBEDDER_MODEL} ...")
        _processor = AutoImageProcessor.from_pretrained(config.EMBEDDER_MODEL)
        _model = AutoModel.from_pretrained(config.EMBEDDER_MODEL)
        _model.eval()
    return _processor, _model


def embed_image(image):
    """image: PIL Image (RGB). Returns a normalized 1D numpy vector."""
    return embed_image_with_raw(image)[0]


def embed_image_with_raw(image):
    """Same forward pass; returns (L2-normalised vector, raw CLS vector)."""
    processor, model = load_embedder()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    raw = outputs.last_hidden_state[:, 0, :].numpy()[0].astype("float32")  # CLS token
    embedding = raw / (np.linalg.norm(raw) + 1e-8)  # L2 normalize, matches catalogue index
    return embedding.astype("float32"), raw