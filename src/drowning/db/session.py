"""Database engine helpers. Connection strings are never logged or returned."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, inspect, text

EXPECTED_SCHEMA_HEAD = "20261026_01"


def normalize_database_url(database_url: str) -> str:
    """Providers hand out ``postgresql://``; this app uses the psycopg 3 driver."""
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url[len("postgresql://"):]
    return database_url


def create_db_engine(database_url: str, *, pool_pre_ping: bool = True) -> Engine:
    if not database_url:
        raise ValueError("DATABASE_URL is not configured")
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("Only PostgreSQL DATABASE_URL values are supported")
    database_url = normalize_database_url(database_url)
    return create_engine(database_url, pool_pre_ping=pool_pre_ping, hide_parameters=True)


def check_db(engine: Engine) -> bool:
    with engine.connect() as connection:
        if connection.execute(text("SELECT 1")).scalar_one() != 1:
            return False
        if "alembic_version" not in inspect(connection).get_table_names():
            return False
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
        return version == EXPECTED_SCHEMA_HEAD
