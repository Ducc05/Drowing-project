"""Render an initial Alembic revision from the current SQLAlchemy metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from alembic.autogenerate.api import produce_migrations, render_python_code
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from drowning.db.models import Base


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "20261026_01_initial_schema.py"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        migration = produce_migrations(context, Base.metadata)
        upgrade_code = render_python_code(migration.upgrade_ops, migration_context=context)
        downgrade_code = render_python_code(migration.downgrade_ops, migration_context=context)
    slot_insert = '''    slot = sa.table("execution_slot", sa.column("slot_id", sa.Integer),
                    sa.column("fencing_token", sa.Integer), sa.column("reconciliation_status", sa.String),
                    sa.column("updated_at", sa.DateTime(timezone=True)))
    op.bulk_insert(slot, [{"slot_id": 1, "fencing_token": 0, "reconciliation_status": "clear",
                           "updated_at": datetime.now(UTC)}])
'''
    upgrade_code = upgrade_code.replace("    op.create_table('job_events'", slot_insert + "    op.create_table('job_events'")
    template = f'''"""Initial shared dataset, model, job, worker and execution-slot schema."""
from datetime import UTC, datetime
from alembic import op
import sqlalchemy as sa

revision = "20261026_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
{upgrade_code}


def downgrade() -> None:
{downgrade_code}
'''
    target.write_text(template, encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
