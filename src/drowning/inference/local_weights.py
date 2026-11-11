"""H20Saver weights the user can choose from: the configured default plus every ``*.pt`` in the models folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from drowning.config import Settings
    from drowning.inference.h20saver import H20SaverIdentity


@dataclass(frozen=True)
class LocalWeights:
    path: Path
    identity: "H20SaverIdentity"
    name: str  # shown to the user: relative to the models folder when inside it, else the file name
    is_default: bool


def _display_name(path: Path, models_dir: Path | None) -> str:
    if models_dir is not None:
        try:
            return path.relative_to(models_dir).as_posix()
        except ValueError:
            pass
    return path.name


def scan_local_weights(models_dir: Path | None, default_weights: Path | None) -> list[LocalWeights]:
    """The default weights first, then every other ``*.pt`` under ``models_dir`` ordered by name.

    A model is its content checksum, so files with identical bytes appear once. Only ``.pt`` files count:
    downloaded archives and unpacked checkpoint folders next to them are ignored.
    """
    from drowning.inference.h20saver import h20saver_identity

    root = models_dir.expanduser().resolve() if models_dir is not None else None
    if root is not None and not root.is_dir():
        root = None
    default = default_weights.expanduser().resolve() if default_weights is not None else None
    if default is not None and not default.is_file():
        default = None

    candidates: list[tuple[Path, bool]] = [(default, True)] if default is not None else []
    if root is not None:
        candidates += [(path, False) for path in sorted(root.rglob("*.pt"), key=lambda p: p.relative_to(root).as_posix())
                       if path.is_file() and path != default]

    found: list[LocalWeights] = []
    seen: set[UUID] = set()
    for path, is_default in candidates:
        identity = h20saver_identity(path)
        if identity.id in seen:
            continue
        seen.add(identity.id)
        found.append(LocalWeights(path=path, identity=identity, name=_display_name(path, root), is_default=is_default))
    return found


def local_weights(settings: "Settings") -> list[LocalWeights]:
    models_dir = Path(settings.h20saver_models_dir) if settings.h20saver_models_dir else None
    default = Path(settings.h20saver_weights_path) if settings.h20saver_weights_path else None
    return scan_local_weights(models_dir, default)


def find_local_weights(settings: "Settings", model_id: UUID) -> LocalWeights | None:
    """The weights file for a model ID. Only files found by the scan can match, so no path is ever taken from a request."""
    return next((item for item in local_weights(settings) if item.identity.id == model_id), None)
