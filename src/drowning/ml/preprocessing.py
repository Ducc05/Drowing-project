"""Shared RGB crop/letterbox/normalization used by image and video inference."""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageOps

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - optional ML environment
    raise ImportError("Install the optional ML dependencies with `pip install -e '.[ml]'`.") from exc


def crop_person(image: Image.Image, bbox_xyxy: Sequence[float], *, context_fraction: float = 0.15,
                output_size: int = 224) -> Image.Image:
    if len(bbox_xyxy) != 4 or context_fraction < 0 or output_size < 16:
        raise ValueError("invalid crop parameters")
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    width, height = oriented.size
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError("bbox must be valid xyxy pixels in the oriented image")
    pad_x, pad_y = (x2 - x1) * context_fraction, (y2 - y1) * context_fraction
    bounds = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
              min(width, int(x2 + pad_x + 0.999)), min(height, int(y2 + pad_y + 0.999)))
    return ImageOps.pad(oriented.crop(bounds), (output_size, output_size),
                        method=Image.Resampling.LANCZOS, color=(114, 114, 114))


def image_to_tensor(image: Image.Image, *, size: int = 224,
                    mean: Sequence[float] = (0.485, 0.456, 0.406),
                    std: Sequence[float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean/std must contain three channels and positive standard deviations")
    rgb = ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), (size, size),
                       method=Image.Resampling.LANCZOS, color=(114, 114, 114))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - torch.tensor(mean, dtype=tensor.dtype)[:, None, None]) / torch.tensor(std, dtype=tensor.dtype)[:, None, None]
