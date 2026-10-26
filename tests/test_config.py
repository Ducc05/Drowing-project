import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from drowning.cli import main
from drowning.config import Settings, load_env_file


class ConfigTests(unittest.TestCase):
    def test_env_file_fills_missing_values_but_preserves_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("DATABASE_URL='postgresql://secret'\nAPI_PORT=9001\n", encoding="utf-8")
            environ = {"API_PORT": "9002"}
            loaded = load_env_file(env_path, environ)
            self.assertEqual(loaded["DATABASE_URL"], "postgresql://secret")
            self.assertEqual(environ["DATABASE_URL"], "postgresql://secret")
            self.assertEqual(environ["API_PORT"], "9002")

    def test_settings_reject_local_training_and_non_kaggle_executor(self):
        with self.assertRaisesRegex(ValueError, "Local training is disabled"):
            Settings.from_env({"ALLOW_LOCAL_TRAIN": "true"})
        with self.assertRaisesRegex(ValueError, "must be kaggle"):
            Settings.from_env({"TRAIN_DEFAULT_EXECUTOR": "local"})

    def test_config_status_never_prints_database_secret(self):
        output = StringIO()
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:passwordsecret@host/db"}, clear=True):
            with redirect_stdout(output):
                result = main(["--env-file", "missing.env", "config-status"])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertNotIn("password", output.getvalue())
        self.assertTrue(payload["database_configured"])
        self.assertFalse(payload["local_training_enabled"])

    def test_config_distinguishes_bundle_path_from_real_approved_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory) / "approved"
            settings = Settings.from_env({"APPROVED_MODEL_BUNDLE_DIR": str(bundle_dir)})
            self.assertTrue(settings.status()["model_bundle_configured"])
            self.assertFalse(settings.status()["model_bundle_manifest_present"])
            bundle_dir.mkdir()
            (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
            settings = Settings.from_env({"APPROVED_MODEL_BUNDLE_DIR": str(bundle_dir)})
            status = settings.status()
            self.assertTrue(status["model_bundle_manifest_present"])
            self.assertNotIn(str(bundle_dir), str(status))


class DatabaseUrlTests(unittest.TestCase):
    def test_unpooled_url_wins_when_both_exist_and_plain_scheme_is_normalized(self):
        from drowning.config import database_url_from
        from drowning.db.session import normalize_database_url
        self.assertEqual(database_url_from({"DATABASE_URL": "postgresql://a", "DATABASE_URL_UNPOOLED": "postgresql://b"}), "postgresql://b")
        self.assertEqual(database_url_from({"DATABASE_URL": "postgresql://a"}), "postgresql://a")
        self.assertEqual(database_url_from({}), "")
        self.assertEqual(normalize_database_url("postgresql://u:p@h/d?sslmode=require"), "postgresql+psycopg://u:p@h/d?sslmode=require")
        self.assertEqual(normalize_database_url("postgresql+psycopg://u:p@h/d"), "postgresql+psycopg://u:p@h/d")


if __name__ == "__main__":
    unittest.main()
