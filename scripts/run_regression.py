"""Run the real local suite and save machine-readable, secret-free outcomes."""
import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def record(self, test, status, reason=None):
        self.rows.append({"test": test.id(), "status": status,
                          "duration_seconds": round(time.perf_counter() - self.started, 4),
                          **({"reason": reason} if reason else {})})

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, "failed", err[0].__name__)

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, "error", err[0].__name__)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        # Class-level skipped fixtures may not call startTest.
        self.started = getattr(self, "started", time.perf_counter())
        self.record(test, "skipped", reason)


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    output = ROOT / "reports" / "regression-results.json"
    output.parent.mkdir(exist_ok=True)
    evidence = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": r".\.venv\Scripts\python scripts\run_regression.py",
                "tests_run": result.testsRun, "passed": sum(r["status"] == "passed" for r in result.rows),
                "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
                "duration_seconds": round(time.perf_counter() - started, 3), "tests": result.rows,
                "real_training_run": False, "live_drive_or_kaggle_run": False}
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evidence: {output}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
