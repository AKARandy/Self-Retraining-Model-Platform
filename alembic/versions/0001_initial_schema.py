"""Initial schema — 7 app tables (mlops DB only).

Revision ID: 0001
Revises: None
Create Date: 2026-09-02

This migration captures the schema from app/core/models.py as of
2026-09-02 (pre-Alembic). It matches Base.metadata.create_all output
and is intended to be `stamp`ed on existing populated databases,
not executed against them. See scripts/baseline-existing-db.sh.

MLflow's `mlflow` database is separate (postgresql://.../mlflow in
infra/docker-compose.yml) and is NOT managed here — no mlflow tables.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("dvc_md5", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=300), nullable=False),
        sa.Column("original_filename", sa.String(length=300), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=True),
        sa.Column("n_cols", sa.Integer(), nullable=True),
        sa.Column("target_column", sa.String(length=100), nullable=True),
        sa.Column("column_stats", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )
    op.create_table(
        "feature_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_versions.id"), nullable=False),
        sa.Column("feature_hash", sa.String(length=64), nullable=False),
        sa.Column("primitives", sa.JSON(), nullable=True),
        sa.Column("artifact_key", sa.String(length=300), nullable=False),
        sa.Column("n_features", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("argo_uid", sa.String(length=64), nullable=True),
        sa.Column("argo_name", sa.String(length=120), nullable=True, unique=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_versions.id"), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("features", sa.JSON(), nullable=True),
        sa.Column("prediction", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "drift_checks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("features_drifted", sa.JSON(), nullable=True),
        sa.Column("report_key", sa.String(length=300), nullable=True),
        sa.Column("triggered_retrain", sa.Boolean(), nullable=False),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("resource", sa.String(length=200), nullable=False),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    # Forward-only in production; downgrade retained for test/CI use.
    op.drop_table("audit_log")
    op.drop_table("drift_checks")
    op.drop_table("predictions")
    op.drop_table("training_runs")
    op.drop_table("feature_sets")
    op.drop_table("dataset_versions")
    op.drop_table("datasets")
