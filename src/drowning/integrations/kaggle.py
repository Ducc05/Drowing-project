"""Official Kaggle CLI wrapper with privacy checks and ambiguous-push handling."""

from __future__ import annotations

import re
import json
import hashlib
import time
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from drowning.logging_utils import redact_secrets

_KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?$")


def validate_checkpoint_dataset(directory: Path, dataset_ref: str, parent: dict) -> None:
    """Only the pinned model files and dataset metadata may be uploaded."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/drown-parent-[a-f0-9]{32}", dataset_ref):
        raise ValueError("invalid private checkpoint dataset reference")
    files = list(directory.rglob("*"))
    expected = {"manifest.json", "weights.pt", "dataset-metadata.json"}
    if {p.name for p in files if p.is_file()} != expected or any(p.is_dir() or p.is_symlink() for p in files):
        raise ValueError("checkpoint dataset may contain only manifest, weights and metadata")
    metadata = json.loads((directory / "dataset-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("id") != dataset_ref:
        raise ValueError("checkpoint dataset identity mismatch")
    for name, key in (("manifest.json", "manifest_sha256"), ("weights.pt", "weights_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != parent[key]:
            raise ValueError("checkpoint dataset checksum mismatch")


def find_kaggle() -> str | None:
    """The kaggle executable on PATH, or the one installed beside this Python (an unactivated venv)."""
    found = shutil.which("kaggle")
    if found:
        return found
    scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        if (scripts / name).is_file():
            return str(scripts / name)
    return None


class KaggleError(RuntimeError):
    pass


class RemoteStateUnconfirmed(KaggleError):
    """The CLI timed out after an operation that may have reached Kaggle."""


@dataclass(frozen=True)
class KaggleCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class KaggleCLIClient:
    def __init__(self, executable: str | None = None, *, timeout_seconds: int = 60,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.executable = executable or find_kaggle() or "kaggle"
        self.timeout_seconds = timeout_seconds
        self.runner = runner

    @staticmethod
    def validate_kernel_ref(kernel_ref: str, *, exact_version: bool = False) -> str:
        if not _KERNEL_REF.fullmatch(kernel_ref):
            raise ValueError("Kaggle kernel ref must be owner/slug[/version]")
        has_version = kernel_ref.rsplit("/", 1)[-1].isdigit() and kernel_ref.count("/") == 2
        if exact_version and not has_version:
            raise ValueError("An exact Kaggle version is required for output collection")
        return kernel_ref

    def _invoke(self, args: Sequence[str], *, timeout_seconds: int | None = None,
                operation_may_have_started: bool = False) -> KaggleCommandResult:
        command = [self.executable, *args]
        try:
            result = self.runner(command, capture_output=True, text=True, timeout=timeout_seconds or self.timeout_seconds,
                                 shell=False, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            if operation_may_have_started:
                raise RemoteStateUnconfirmed("Kaggle command timed out; reconcile kernel status before retrying") from exc
            raise KaggleError("Kaggle CLI command timed out") from exc
        except FileNotFoundError as exc:
            raise KaggleError("Kaggle CLI executable was not found in PATH") from exc
        response = KaggleCommandResult(tuple(command), result.returncode,
                                       redact_secrets(result.stdout or ""), redact_secrets(result.stderr or ""))
        if result.returncode:
            raise KaggleError(f"Kaggle CLI exited {result.returncode}: {response.stderr[-1000:]}")
        return response

    def version(self) -> KaggleCommandResult:
        return self._invoke(["--version"])

    def status(self, kernel_ref: str) -> KaggleCommandResult:
        self.validate_kernel_ref(kernel_ref)
        return self._invoke(["kernels", "status", kernel_ref])

    def dataset_status(self, dataset_ref: str) -> KaggleCommandResult:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", dataset_ref):
            raise ValueError("Kaggle dataset ref must be owner/slug")
        return self._invoke(["datasets", "status", dataset_ref])

    def push_private_checkpoint_dataset(self, directory: Path, dataset_ref: str, parent: dict, *,
                                        sleep: Callable[[float], None] = time.sleep) -> KaggleCommandResult:
        validate_checkpoint_dataset(directory, dataset_ref, parent)
        # No --public flag: the official CLI creates a private dataset by default.
        result = self._invoke(["datasets", "create", "-p", str(directory.resolve()), "-q"],
                              timeout_seconds=300, operation_may_have_started=True)
        for attempt in range(12):
            status = self._invoke(["datasets", "status", dataset_ref], timeout_seconds=15).stdout.strip().strip('"').casefold()
            if status in {"ready", "datasetstatus.ready"}:
                return result
            if status in {"error", "failed", "datasetstatus.error"}:
                raise KaggleError("Private checkpoint dataset processing failed; kernel was not pushed")
            if attempt < 11:
                sleep(5)
        raise KaggleError("Private checkpoint dataset did not become ready; kernel was not pushed")

    def push_private_kernel(self, kernel_dir: Path, *, timeout_seconds: int = 3600,
                            no_run: bool = False) -> KaggleCommandResult:
        metadata_path = kernel_dir / "kernel-metadata.json"
        if not kernel_dir.is_dir() or not metadata_path.is_file():
            raise ValueError("kernel directory must contain kernel-metadata.json")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("is_private") is not True:
            raise ValueError("Refusing to push a Kaggle kernel unless is_private is true")
        kernel_ref = self.validate_kernel_ref(str(metadata.get("id", "")))
        code_file = metadata.get("code_file")
        if not isinstance(code_file, str):
            raise ValueError("kernel-metadata.json requires code_file")
        resolved_file = (kernel_dir / code_file).resolve()
        try:
            resolved_file.relative_to(kernel_dir.resolve())
        except ValueError as exc:
            raise ValueError("code_file must remain inside the kernel directory") from exc
        if not resolved_file.is_file():
            raise ValueError("kernel code_file does not exist")
        allowed_files = {metadata_path.resolve(), resolved_file}
        extra_files = [path for path in kernel_dir.rglob("*") if path.is_file() and path.resolve() not in allowed_files]
        if extra_files:
            raise ValueError("Kaggle kernel directory may contain only its notebook and kernel-metadata.json")
        args = ["kernels", "push", "-p", str(kernel_dir.resolve()), "-t", str(timeout_seconds)]
        if no_run:
            args.append("--no-run")
        return self._invoke(args, timeout_seconds=timeout_seconds + 30, operation_may_have_started=True)

    def collect_exact_output(self, kernel_ref: str, destination: Path) -> KaggleCommandResult:
        self.validate_kernel_ref(kernel_ref, exact_version=True)
        target = destination.resolve()
        try:
            target.relative_to(Path("D:/DrowningDemo").resolve())
            under_demo = True
        except ValueError:
            under_demo = False
        if under_demo or any(part.casefold() == "sources" for part in target.parts):
            raise ValueError("Protected D:\\DrowningDemo or sources/ output path")
        if target.exists() and any(target.iterdir()):
            raise ValueError("Output destination must be absent or empty; refusing to overwrite artifacts")
        target.mkdir(parents=True, exist_ok=True)
        return self._invoke(["kernels", "output", kernel_ref, "-p", str(target)])
