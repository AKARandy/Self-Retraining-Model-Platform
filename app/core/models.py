from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    version: Mapped[int] = mapped_column(Integer)
    dvc_md5: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(300))
    original_filename: Mapped[str] = mapped_column(String(300))
    n_rows: Mapped[int | None] = mapped_column(nullable=True)
    n_cols: Mapped[int | None] = mapped_column(nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(100), nullable=True)
    column_stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    feature_hash: Mapped[str] = mapped_column(String(64))
    primitives: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_key: Mapped[str] = mapped_column(String(300))
    n_features: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    argo_uid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    argo_name: Mapped[str | None] = mapped_column(String(120), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="submitted")
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), default="live")  # live | batch
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prediction: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftCheck(Base):
    __tablename__ = "drift_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    verdict: Mapped[str] = mapped_column(String(16))  # ok | drift
    features_drifted: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_key: Mapped[str | None] = mapped_column(String(300), nullable=True)
    triggered_retrain: Mapped[bool] = mapped_column(Boolean, default=False)
    training_run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(String(80))
    resource: Mapped[str] = mapped_column(String(200))
    allowed: Mapped[bool] = mapped_column(Boolean)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
