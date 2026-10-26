"""Per-installation identity and process liveness helpers for crash recovery."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def installation_id(state_dir: Path) -> str:
    """Stable random ID for this checkout/machine, created on first use under ``state_dir``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "installation-id"
    try:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) == 32 and all(c in "0123456789abcdef" for c in value):
            return value
    except FileNotFoundError:
        pass
    value = uuid.uuid4().hex
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    except FileExistsError:  # another process won the race
        return path.read_text(encoding="utf-8").strip()
    return value


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information, still_active = 0x1000, 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


DEFAULT_WORKER_ID = "local-worker"


def resolve_worker_id(configured: str, input_cache_dir: str) -> str:
    """Use WORKER_ID when set; otherwise a stable per-installation name so two machines never collide."""
    if configured and configured != DEFAULT_WORKER_ID:
        return configured
    return f"host-{installation_id(Path(input_cache_dir).expanduser().resolve().parent)[:12]}"
