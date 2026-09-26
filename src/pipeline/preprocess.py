"""
Phase 1 (pipeline stage) — image preprocessing.
EXIF orientation fix (critical for phone photos — many phones write
rotation as metadata rather than actually rotating pixels), resize
cap for consistent downstream processing, basic validity check.
"""

from PIL import Image, ImageOps
import sys
sys.path.append(".")
import config

MAX_DIMENSION = 1600  # cap long edge — keeps detection/embedding fast, no meaningful quality loss beyond this


def preprocess_image(image):
    """
    image: PIL Image
    Returns: PIL Image, RGB, EXIF-rotated correctly, resized if needed.
    Raises ValueError if the image is invalid/unreadable.
    """
    if image is None:
        raise ValueError("No image provided")

    # Fix EXIF rotation — phone photos often store rotation as metadata
    # rather than physically rotating pixels; without this, crops/boxes
    # end up on a sideways image.
    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    if image.size[0] < 10 or image.size[1] < 10:
        raise ValueError(f"Image too small after processing: {image.size}")

    return image


def load_and_preprocess(path):
    """Convenience: load from disk path and preprocess in one call."""
    image = Image.open(path)
    return preprocess_image(image)