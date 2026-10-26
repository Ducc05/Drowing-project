"""Provision an isolated loopback PostgreSQL development/test container when .env DB fields are empty."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from pathlib import Path

from drowning.config import load_env_file

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "drowning-local-pg-20261026"
VOLUME = "drowning-local-pgdata-20261026"
PORT = 55439


def docker(*args, env=None, check=True):
    return subprocess.run(["docker", *args], env=env, capture_output=True, text=True, check=check, timeout=45)


def main():
    env_path = ROOT / ".env"
    values = {}
    load_env_file(env_path, values)
    if values.get("DATABASE_URL") or values.get("TEST_DATABASE_URL"):
        print(json.dumps({"action": "skipped", "reason": "A DB URL is already configured; no existing value was replaced."}))
        return 0
    existing = docker("container", "ls", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}")
    resumed = bool(existing.stdout.strip())
    if resumed:
        owned = json.loads(docker("inspect", CONTAINER).stdout)[0]
        if (owned["Config"].get("Labels", {}).get("org.drowning.project") != "local-review"
                or not any(mount.get("Name") == VOLUME for mount in owned.get("Mounts", []))):
            raise RuntimeError("Existing container is not owned by this provisioner; no configuration was changed.")
        container_env = dict(entry.split("=", 1) for entry in owned["Config"]["Env"] if "=" in entry)
        password = container_env["POSTGRES_PASSWORD"]
        image = owned["Image"]
        if not owned["State"]["Running"]:
            docker("start", CONTAINER)
    else:
        volume = docker("volume", "ls", "--filter", f"name=^{VOLUME}$", "--format", "{{.Name}}")
        if volume.stdout.strip():
            raise RuntimeError("The dedicated volume name already exists; no existing database files were reused.")
        image = docker("image", "inspect", "postgres:17", "--format", "{{.Id}}").stdout.strip()
        password = secrets.token_urlsafe(32)
        runtime_env = {**os.environ, "POSTGRES_PASSWORD": password}
        docker("run", "-d", "--name", CONTAINER, "--label", "org.drowning.project=local-review",
               "--restart", "unless-stopped", "-p", f"127.0.0.1:{PORT}:5432",
               "--env", "POSTGRES_PASSWORD", "--env", "POSTGRES_USER=drowning",
               "--env", "POSTGRES_DB=drowning_dev", "--mount", f"type=volume,src={VOLUME},dst=/var/lib/postgresql/data",
               image, env=runtime_env)
    ready = False
    for _ in range(30):
        probe = docker("exec", CONTAINER, "pg_isready", "-h", "127.0.0.1", "-U", "drowning", "-d", "drowning_dev", check=False)
        if probe.returncode == 0:
            ready = True
            break
        time.sleep(1)
    if not ready:
        raise RuntimeError("Dedicated PostgreSQL startup did not become ready; .env was not changed.")
    test_exists = docker("exec", CONTAINER, "psql", "-U", "drowning", "-d", "postgres", "-At",
                         "-c", "SELECT 1 FROM pg_database WHERE datname='drowning_test'").stdout.strip()
    if not test_exists:
        docker("exec", CONTAINER, "psql", "-U", "drowning", "-d", "postgres", "-v", "ON_ERROR_STOP=1",
               "-c", "CREATE DATABASE drowning_test")
    version = docker("exec", CONTAINER, "psql", "-U", "drowning", "-d", "drowning_dev", "-At",
                     "-c", "SHOW server_version").stdout.strip()
    base = f"postgresql+psycopg://drowning:{password}@127.0.0.1:{PORT}/"
    content = env_path.read_text(encoding="utf-8")
    lines = []
    seen = set()
    new_values = {"DATABASE_URL": base + "drowning_dev", "TEST_DATABASE_URL": base + "drowning_test"}
    for line in content.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key in new_values:
            lines.append(f"{key}={new_values[key]}")
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in new_values.items() if key not in seen)
    temp = env_path.with_name(".env.postgres-setup.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(env_path)
    evidence = {"action": "resumed_owned_container" if resumed else "created", "container": CONTAINER, "image_id": image, "postgres_version": version,
                "bind": "127.0.0.1", "port": PORT, "development_database": "drowning_dev",
                "test_database": "drowning_test", "credentials": "stored only in ignored local .env",
                "shared_cloud_database_verified": False}
    report = ROOT / "local-cache" / "review" / "postgres_setup.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
