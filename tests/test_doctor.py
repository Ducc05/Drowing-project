import json
import tempfile
import unittest
from pathlib import Path

from drowning.cli import main
from drowning.doctor import run_doctor, summarize
from drowning.integrations.drive import DriveError, authorize_drive, check_folder_access

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "s3cret-pass-word"


def doctor(env, **kwargs):
    kwargs.setdefault("which", lambda name: None)
    return {c.id: c for c in run_doctor(env, project_root=ROOT, home=Path(tempfile.gettempdir()) / "no-home", **kwargs)}


class DoctorTests(unittest.TestCase):
    def test_empty_environment_lists_every_required_item_without_values(self):
        checks = doctor({})
        for key in ("database_url", "kaggle_cli", "kaggle_credentials", "kaggle_kernel_ref", "kaggle_datasets"):
            self.assertFalse(checks[key].ok, key)
            self.assertTrue(checks[key].required, key)
        self.assertTrue(checks["kaggle_private"].ok)
        self.assertIn("required_missing", summarize(list(checks.values())))

    def test_fully_configured_environment_is_ready_and_never_echoes_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.json"
            token.write_text("{}", encoding="utf-8")
            env = {"DATABASE_URL": f"postgresql+psycopg://user:{PASSWORD}@db.example.com:5432/drown?sslmode=require",
                   "TEST_DATABASE_URL": f"postgresql+psycopg://user:{PASSWORD}@localhost:5432/drown_test",
                   "KAGGLE_USERNAME": "me", "KAGGLE_KEY": "key-value-123", "KAGGLE_KERNEL_REF": "me/drowning-temporal",
                   "KAGGLE_DATASET_SOURCES": "me/pool-images,me/pool-videos", "GOOGLE_DRIVE_FOLDER_ID": "folder-id-value",
                   "GOOGLE_OAUTH_TOKEN_FILE": str(token)}
            checks = run_doctor(env, project_root=ROOT, which=lambda name: "kaggle", home=Path(directory))
            report = summarize(checks)
            self.assertTrue(report["ready_for_kaggle_runs"], report["required_missing"])
            text = json.dumps(report, ensure_ascii=False)
            for secret in (PASSWORD, "key-value-123", "folder-id-value"):
                self.assertNotIn(secret, text)

    def test_remote_database_without_sslmode_and_shared_test_database_are_flagged(self):
        env = {"DATABASE_URL": "postgresql+psycopg://u:p@db.example.com/drown",
               "TEST_DATABASE_URL": "postgresql+psycopg://u:p@db.example.com/drown"}
        checks = doctor(env)
        self.assertFalse(checks["database_ssl"].ok)
        self.assertFalse(checks["test_database_separate"].ok)
        self.assertFalse(doctor({"DATABASE_URL": "sqlite:///x.db"})["database_is_postgresql"].ok)

    def test_malformed_references_are_rejected(self):
        checks = doctor({"KAGGLE_KERNEL_REF": "no-slash", "KAGGLE_DATASET_SOURCES": "a/b,bad"})
        self.assertFalse(checks["kaggle_kernel_ref"].ok)
        self.assertFalse(checks["kaggle_datasets"].ok)

    def test_online_checks_use_injected_probes_and_report_failures_plainly(self):
        env = {"DATABASE_URL": "postgresql+psycopg://u:p@localhost/d", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k",
               "KAGGLE_DATASET_SOURCES": "me/a,me/b"}
        checks = doctor(env, online=True, which=lambda n: "kaggle", db_check=lambda url: False,
                        kaggle_check=lambda ref: None if ref == "me/a" else "Dataset me/b: not found")
        self.assertFalse(checks["database_online"].ok)
        self.assertTrue(checks["kaggle_dataset_online:me/a"].ok)
        self.assertFalse(checks["kaggle_dataset_online:me/b"].ok)

        def boom(url):
            raise ConnectionError(url)
        message = doctor(env, online=True, db_check=boom)["database_online"].message
        self.assertIn("ConnectionError", message)
        self.assertNotIn("postgresql", message)

    def test_cli_doctor_exit_code_and_json_output(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--env-file", str(ROOT / "does-not-exist.env"), "doctor", "--json"])
        self.assertIn(code, (0, 3))
        self.assertIn("checks", json.loads(buffer.getvalue()))


class DriveAuthTests(unittest.TestCase):
    class FakeCredentials:
        refresh_token = "r"

        def to_json(self):
            return '{"token": "t"}'

    class FakeFlow:
        def run_local_server(self, port):
            return DriveAuthTests.FakeCredentials()

    def test_authorize_writes_token_once_and_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            secret, token = base / "client.json", base / "secrets" / "token.json"
            secret.write_text("{}", encoding="utf-8")
            result = authorize_drive(secret, token, flow_factory=lambda path, scopes: self.FakeFlow())
            self.assertTrue(result["token_file_written"] and result["has_refresh_token"])
            self.assertEqual(set(result), {"token_file_written", "scope", "has_refresh_token"})
            self.assertTrue(token.is_file())
            with self.assertRaises(DriveError):
                authorize_drive(secret, token, flow_factory=lambda path, scopes: self.FakeFlow())
            authorize_drive(secret, token, force=True, flow_factory=lambda path, scopes: self.FakeFlow())
            with self.assertRaises(DriveError):
                authorize_drive(base / "missing.json", base / "other.json")

    def test_folder_check_requires_a_folder_the_account_can_write_to(self):
        def service(meta=None, error=None):
            class Request:
                def execute(self_inner):
                    if error:
                        raise error
                    return meta

            class Files:
                def get(self_inner, **kwargs):
                    return Request()

            class Service:
                def files(self_inner):
                    return Files()
            return Service()
        folder = "application/vnd.google-apps.folder"
        self.assertIsNone(check_folder_access(service({"mimeType": folder, "capabilities": {"canAddChildren": True}}), "x"))
        self.assertIn("quyền", check_folder_access(service({"mimeType": folder, "capabilities": {"canAddChildren": False}}), "x"))
        self.assertIn("không phải", check_folder_access(service({"mimeType": "text/plain"}), "x"))
        self.assertIn("RuntimeError", check_folder_access(service(error=RuntimeError("403")), "x"))


class DetectorFetchTests(unittest.TestCase):
    def spec(self, payload=b"fake-weights"):
        import hashlib
        return {"name": "d.pt", "url": "https://example.invalid/d.pt", "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    def opener(self, payload):
        import io

        class Response(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False
        return lambda url, timeout=0: Response(payload)

    def test_download_is_verified_atomic_and_idempotent(self):
        from drowning.detector_fetch import fetch_detector
        with tempfile.TemporaryDirectory() as directory:
            first = fetch_detector(Path(directory), spec=self.spec(), opener=self.opener(b"fake-weights"))
            self.assertTrue(first["downloaded"])
            second = fetch_detector(Path(directory), spec=self.spec(), opener=self.opener(b"WRONG"))
            self.assertFalse(second["downloaded"])  # already valid: the network is not used again
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_wrong_bytes_are_discarded_and_foreign_files_are_never_overwritten(self):
        from drowning.detector_fetch import DetectorFetchError, fetch_detector
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DetectorFetchError):
                fetch_detector(Path(directory), spec=self.spec(), opener=self.opener(b"tampered!!!!"))
            self.assertEqual(list(Path(directory).iterdir()), [])
            (Path(directory) / "d.pt").write_bytes(b"another file")
            with self.assertRaises(DetectorFetchError):
                fetch_detector(Path(directory), spec=self.spec(), opener=self.opener(b"fake-weights"))
            self.assertEqual((Path(directory) / "d.pt").read_bytes(), b"another file")

    def test_env_update_touches_only_the_empty_key_and_keeps_line_endings(self):
        from drowning.detector_fetch import set_env_value
        crlf = bytes([13, 10])
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_bytes(b"A=1" + crlf + b"PERSON_DETECTOR_WEIGHTS=" + crlf + b"B=2" + crlf)
            self.assertTrue(set_env_value(env, "PERSON_DETECTOR_WEIGHTS", "X:/m.pt"))
            self.assertEqual(env.read_bytes(), b"A=1" + crlf + b"PERSON_DETECTOR_WEIGHTS=X:/m.pt" + crlf + b"B=2" + crlf)
            self.assertFalse(set_env_value(env, "PERSON_DETECTOR_WEIGHTS", "Y"))
            self.assertIn(b"X:/m.pt", env.read_bytes())


if __name__ == "__main__":
    unittest.main()
