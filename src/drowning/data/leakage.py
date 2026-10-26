"""How many held-out images have a near copy in the training images (flips and 90-degree turns included).

A split by group only keeps identical or near-identical images together when the hash says so, and a plain
difference hash misses a mirrored copy (a common export augmentation). This report matches each held-out image
against the training images under the eight flips/turns of the square. It is a LOWER bound: a copy that was
cropped, tilted by another angle or recoloured strongly is not found, so a low number is not proof of no leak.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_TRANSFORMS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
               Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
               Image.Transpose.TRANSVERSE)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dhash_image(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, :-1] > pixels[:, 1:]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def identity_dhash(path: Path) -> int:
    with Image.open(path) as opened:
        return _dhash_image(opened.convert("RGB"))


def dihedral_dhashes(path: Path) -> list[int]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    return [_dhash_image(image if t is None else image.transpose(t)) for t in _TRANSFORMS]


def _popcount(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    flat = _POPCOUNT_TABLE[values.view(np.uint8).reshape(*values.shape, 8)]
    return flat.sum(axis=-1)


def near_copy_report(reference_paths: list[Path], query_paths: list[Path], *, max_distance: int = 4,
                     workers: int = 8) -> dict[str, Any]:
    """Fraction of ``query_paths`` with a reference image within ``max_distance`` bits under flips and turns."""
    if not reference_paths or not query_paths:
        return {"queries": len(query_paths), "references": len(reference_paths), "with_near_copy": 0, "fraction": None,
                "max_distance": max_distance, "method": "dhash64 under 8 flips/turns", "lower_bound": True}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="leak") as pool:
        references = np.array(list(pool.map(identity_dhash, reference_paths)), dtype=np.uint64)
        queries = np.array(list(pool.map(dihedral_dhashes, query_paths)), dtype=np.uint64)  # (M, 8)
    hits = 0
    for start in range(0, len(queries), 64):
        chunk = queries[start:start + 64]                                  # (c, 8)
        distance = _popcount(chunk[:, :, None] ^ references[None, None, :])  # (c, 8, N)
        hits += int((distance.min(axis=(1, 2)) <= max_distance).sum())
    return {"queries": len(query_paths), "references": len(reference_paths), "with_near_copy": hits,
            "fraction": round(hits / len(query_paths), 4), "max_distance": max_distance,
            "method": "dhash64 under 8 flips/turns", "lower_bound": True}
