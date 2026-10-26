import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


ROOT = Path(__file__).resolve().parents[1]


class MigrationSmokeTests(unittest.TestCase):
    def test_initial_migration_round_trip_on_temporary_sqlite_schema_only(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "migration-smoke.sqlite"
            url = f"sqlite:///{db_path.as_posix()}"
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            with patch.dict(os.environ, {"DATABASE_URL": url}):
                command.upgrade(config, "head")
            engine = create_engine(url)
            try:
                tables = set(inspect(engine).get_table_names())
                self.assertIn("jobs", tables)
                self.assertIn("execution_slot", tables)
                self.assertIn("model_registry", tables)
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text("SELECT COUNT(*) FROM execution_slot WHERE slot_id=1")).scalar_one(), 1)
            finally:
                engine.dispose()
            with patch.dict(os.environ, {"DATABASE_URL": url}):
                command.downgrade(config, "base")
            engine = create_engine(url)
            try:
                self.assertNotIn("jobs", set(inspect(engine).get_table_names()))
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
