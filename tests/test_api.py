import asyncio
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.db.models import Base, ExecutionSlot, Job


ROOT = Path(__file__).resolve().parents[1]


def api_request(app, method: str, path: str, **kwargs):
    async def _request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test.local") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(_request())


class ApiTests(unittest.TestCase):
    def test_health_and_config_status_work_without_database_and_do_not_expose_secrets(self):
        settings = Settings.from_env({})
        app = create_app(settings=settings, config_dir=ROOT / "configs" / "datasets",
                         env_file=ROOT / "missing.env")
        self.assertEqual(api_request(app, "GET", "/health").status_code, 200)
        readiness = api_request(app, "GET", "/ready")
        self.assertEqual(readiness.status_code, 503)
        config = api_request(app, "GET", "/api/v1/config-status").json()
        self.assertFalse(config["database_configured"])
        self.assertNotIn("DATABASE_URL", config)
        sources = api_request(app, "GET", "/api/v1/sources").json()["sources"]
        alanoud = next(item for item in sources if item["source_id"] == "alanoud_swimming_drowning")
        self.assertEqual(alanoud["mapping_review_status"], "pending_sample_review")
        self.assertNotIn("D:/DrowningKaggleCheck/data", str(sources))
        self.assertEqual(api_request(app, "GET", "/api/v1/jobs").status_code, 503)

    def test_job_http_flow_uses_idempotency_and_event_history(self):
        engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version (version_num) VALUES ('20261026_01')"))
        try:
            app = create_app(settings=Settings.from_env({}), engine=engine,
                             config_dir=ROOT / "configs" / "datasets", env_file=ROOT / "missing.env")
            body = {"job_type": "prepare", "idempotency_key": "api-request-1234", "payload": {"mode": "prepare_only"}}
            first = api_request(app, "POST", "/api/v1/jobs", json=body)
            second = api_request(app, "POST", "/api/v1/jobs", json={**body, "payload": {"mode": "train_candidate"}})
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            first_payload, second_payload = first.json(), second.json()
            self.assertTrue(first_payload["created"])
            self.assertFalse(second_payload["created"])
            self.assertEqual(first_payload["job"]["id"], second_payload["job"]["id"])
            with Session(engine) as session:
                persisted = session.get(Job, UUID(first_payload["job"]["id"]))
                self.assertEqual(persisted.payload, {"mode": "prepare_only"})
            events = api_request(app, "GET", f"/api/v1/jobs/{first_payload['job']['id']}/events")
            self.assertEqual(events.status_code, 200)
            self.assertEqual(events.json()["events"][0]["to_state"], "queued")
            self.assertEqual(api_request(app, "GET", "/ready").status_code, 200)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
