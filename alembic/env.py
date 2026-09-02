"""Alembic env — owns the app `mlops` database only.

- Loads `Base.metadata` from `app.core.models` (imports models to register tables).
- Reads `DATABASE_URL` from env in every environment and overrides `alembic.ini`.
- Defense-in-depth `include_object` filter: logs if a non-app object appears;
  current arch uses separate Postgres databases (`mlops` vs `mlflow`), so
  a connection to `mlops` structurally cannot see `mlflow` tables — the filter
  is a no-op in that arch and becomes active only if MLflow were moved into
  the same database.
"""
import logging
import os
from logging.config import fileConfig

from sqlalchemy import pool

from alembic import context

# Alembic Config object
config = context.config

# Logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override URL from env (every environment uses DATABASE_URL)
db_url = os.getenv("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# Import models to populate Base.metadata — do NOT call create_all
import app.core.models  # noqa: F401 — side-effect: register tables on Base
from app.core.db import Base

target_metadata = Base.metadata

logger = logging.getLogger("alembic.env")


def include_object(object, name, type_, reflected, compare_to):
    """Defense-in-depth: allow only app tables; log anything unexpected."""
    # Allow all types that belong to target_metadata; for autogenerate,
    # reflected objects not in target_metadata would be dropped.
    # Since we use separate databases, this is a no-op — kept explicit.
    if type_ == "table":
        if name in target_metadata.tables:
            return True
        # Reflected table not in our metadata — would be from another app/db.
        # Let autogenerate surface it (so check fails visibly) rather than silently hide.
        logger.warning("include_object: reflected table %r not in Base.metadata — allowing to surface", name)
        return True
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine

    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
        pool_pre_ping=True,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
