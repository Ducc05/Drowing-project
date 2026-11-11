"""Cell-based confirmation of drowning alarms.

A per-track alarm alone also fires for a rescuer whom the detector mislabels as drowning while they swim fast:
the tracker keeps their ID, so their votes pile up. A person who is really drowning stays roughly in place, so
the frame is divided into a grid and each cell keeps its own time window of drowning votes. A track's alarm is
only confirmed while the cell(s) it occupies hold sustained drowning evidence; someone sweeping across many
cells gives each cell only a short burst and is never confirmed. The gate can only remove alarms, never add them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # keep this module (used by config parsing) free of NumPy
    from drowning.inference.tracking import Box

MAX_CELLS_PER_AXIS = 32
_OFF = {"off", "none", "false", "0", "no", "disabled"}


def parse_cell_grid(text: str) -> tuple[int, int] | None:
    """Parse ``"6x4"`` (columns x rows). ``off`` disables the gate; anything else invalid raises."""
    value = text.strip().casefold()
    if value in _OFF:
        return None
    parts = [part.strip() for part in value.split("x")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("cell grid must look like 6x4 (columns x rows) or off")
    cols, rows = int(parts[0]), int(parts[1])
    if not (1 <= cols <= MAX_CELLS_PER_AXIS and 1 <= rows <= MAX_CELLS_PER_AXIS):
        raise ValueError(f"cell grid sides must be between 1 and {MAX_CELLS_PER_AXIS}")
    return cols, rows


def format_cell_grid(grid: tuple[int, int] | None) -> str:
    """Inverse of ``parse_cell_grid``: ``"6x4"`` or ``"off"``."""
    return "off" if grid is None else f"{grid[0]}x{grid[1]}"


@dataclass(frozen=True)
class CellGrid:
    cols: int
    rows: int
    # A box centre within this fraction of a cell from an edge also counts for the neighbouring cell, so a
    # person resting on a boundary (or jittering across it) does not split their votes below the alarm ratio.
    margin: float = 0.25

    def __post_init__(self) -> None:
        if self.cols < 1 or self.rows < 1:
            raise ValueError("cell grid needs at least one column and one row")
        if not 0 <= self.margin < 0.5:
            raise ValueError("cell margin must be in [0, 0.5)")

    def all_cells(self) -> list[tuple[int, int]]:
        return [(col, row) for row in range(self.rows) for col in range(self.cols)]

    def cells_for_box(self, box: Box, width: float, height: float) -> frozenset[tuple[int, int]]:
        """Cells that hold this person: every cell the box covers, and at least the centre +- ``margin``.

        Using the whole box (not only its centre) keeps the gate meaningful at any distance: a person close to the
        camera covers many cells and a small swing of their centre must not look like leaving the cell, while a
        small swimmer far away occupies about one cell, so crossing the view still spreads thin across cells.
        """
        u0, u1 = box.x1 / width * self.cols, box.x2 / width * self.cols
        v0, v1 = box.y1 / height * self.rows, box.y2 / height * self.rows
        u, v = (u0 + u1) / 2, (v0 + v1) / 2
        cols = range(max(0, min(math.floor(u - self.margin), math.floor(u0))),
                     min(self.cols - 1, max(math.floor(u + self.margin), math.floor(u1))) + 1)
        rows = range(max(0, min(math.floor(v - self.margin), math.floor(v0))),
                     min(self.rows - 1, max(math.floor(v + self.margin), math.floor(v1))) + 1)
        cells = frozenset((c, r) for c in cols for r in rows)
        if cells:
            return cells
        # box entirely outside the frame: snap to the nearest edge cell
        return frozenset({(min(self.cols - 1, max(0, math.floor(u))), min(self.rows - 1, max(0, math.floor(v))))})


class CellGate:
    """Confirms per-track alarms against per-cell drowning vote windows.

    ``monitor`` is a vote monitor exposing ``update(clock, ids, votes)`` and ``alarmed`` (the same
    ``DrowningMonitor``/``LiveDrowningMonitor`` used per track), here keyed by ``(col, row)``.
    """

    def __init__(self, grid: CellGrid, monitor: Any):
        self.grid = grid
        self.monitor = monitor
        self._cells = grid.all_cells()
        self._last_cells: dict[str, frozenset[tuple[int, int]]] = {}

    @property
    def alarmed_cells(self) -> set[tuple[int, int]]:
        return set(self.monitor.alarmed)

    def update(self, clock: float, frame_size: tuple[float, float],
               detections: Iterable[tuple[str, Box, bool]], alarmed_tracks: Iterable[str]) -> set[str]:
        """Fold one frame in; return the subset of ``alarmed_tracks`` whose cells are confirmed."""
        width, height = frame_size
        alarmed_tracks = set(alarmed_tracks)
        voted: set[tuple[int, int]] = set()
        seen: set[str] = set()
        for track_id, box, vote in detections:
            cells = self.grid.cells_for_box(box, width, height)
            self._last_cells[track_id] = cells
            seen.add(track_id)
            if vote:
                voted |= cells
        # Every cell advances every frame so quiet cells drain their votes (and an alarm can clear).
        self.monitor.update(clock, self._cells, [cell in voted for cell in self._cells])
        for track_id in list(self._last_cells):
            if track_id not in seen and track_id not in alarmed_tracks:
                del self._last_cells[track_id]
        hot = self.monitor.alarmed
        return {track_id for track_id in alarmed_tracks if self._last_cells.get(track_id, frozenset()) & hot}
