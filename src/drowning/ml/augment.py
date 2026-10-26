"""Training-time augmentation for person crops. Used on the train split only; deterministic per (seed, epoch, index).

The crops are already cut around a labeled box, so these transforms imitate what differs between the training
photos and a real camera: a slightly wrong box (shift/scale), a person that is further away (zoom out onto grey),
a tilted horizon, other light and colour, and blur or low resolution. Nothing here is a claim about accuracy.
"""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

FILL = (114, 114, 114)  # same grey as the letterbox padding used when the crops were built


def augment_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    img = image.convert("RGB")
    width, height = img.size
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    # A box that is a bit too tight, loose or off-centre: crop a random sub-window and scale it back.
    scale, ratio = rng.uniform(0.70, 1.0), rng.uniform(0.85, 1.15)
    crop_w, crop_h = min(width, int(width * scale * ratio ** 0.5)), min(height, int(height * scale / ratio ** 0.5))
    left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BILINEAR)
    # A person further from the camera: shrink onto the grey canvas.
    if rng.random() < 0.35:
        factor = rng.uniform(0.5, 0.9)
        small = img.resize((max(8, int(width * factor)), max(8, int(height * factor))), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), FILL)
        canvas.paste(small, (rng.randint(0, width - small.width), rng.randint(0, height - small.height)))
        img = canvas
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR, fillcolor=FILL)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        if rng.random() < 0.8:
            img = enhancer(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    if rng.random() < 0.4:  # low resolution: down then up
        factor = rng.uniform(0.25, 0.6)
        small = img.resize((max(8, int(width * factor)), max(8, int(height * factor))), Image.Resampling.BILINEAR)
        img = small.resize((width, height), Image.Resampling.BILINEAR)
    if rng.random() < 0.25:  # random erasing: a small patch, as if hidden by water or another swimmer
        area = rng.uniform(0.02, 0.12) * width * height
        patch_w = max(4, min(width, int((area * rng.uniform(0.5, 2.0)) ** 0.5)))
        patch_h = max(4, min(height, int(area / patch_w)))
        x, y = rng.randint(0, width - patch_w), rng.randint(0, height - patch_h)
        img.paste(Image.new("RGB", (patch_w, patch_h), FILL), (x, y))
    return img


def sample_rng(seed: int, epoch: int, index: int) -> random.Random:
    """Stable across runs and machines (string seeding hashes with SHA-512)."""
    return random.Random(f"augment:{seed}:{epoch}:{index}")
