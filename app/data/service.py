import io
from pathlib import Path

import pandas as pd
from sqlalchemy import func as safunc
from sqlalchemy.orm import Session

from ..core.models import Dataset, DatasetVersion
from . import dvc_io

RAW_DIR = dvc_io.REPO / "data" / "raw"


def column_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        s = df[col]
        entry = {"dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 2)}
        if pd.api.types.is_numeric_dtype(s):
            entry.update(
                mean=None if s.dropna().empty else round(float(s.mean()), 4),
                std=None if s.dropna().empty else round(float(s.std()), 4),
                min=None if s.dropna().empty else float(s.min()),
                max=None if s.dropna().empty else float(s.max()),
            )
        else:
            entry["n_unique"] = int(s.nunique(dropna=True))
        stats[col] = entry
    return stats


def upload_version(
    db: Session,
    name: str,
    filename: str,
    content: bytes,
    target_column: str | None = None,
) -> DatasetVersion:
    import re

    # Simple name validation — prevent path traversal and DB errors
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,100}", name):
        raise ValueError("invalid dataset name (alphanumeric, . _ - , max 100)")
    if target_column and len(target_column) > 100:
        raise ValueError("target_column too long")

    ds = db.query(Dataset).filter_by(name=name).first()
    if ds is None:
        ds = Dataset(name=name)
        db.add(ds)
        db.flush()

    next_v = (db.query(safunc.max(DatasetVersion.version)).filter_by(dataset_id=ds.id).scalar() or 0) + 1

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"invalid CSV: {e}")
    if df.empty:
        raise ValueError("empty dataset")
    if df.shape[1] > 500 or df.shape[0] > 1_000_000:
        raise ValueError("dataset too large")
    if target_column and target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not in columns")
    suffix = Path(filename).suffix or ".csv"
    if suffix not in (".csv", ".txt"):
        suffix = ".csv"
    rel = Path("data/raw") / name / f"v{next_v}{suffix}"
    dest = (dvc_io.REPO / rel).resolve()
    base = (dvc_io.REPO / "data" / "raw").resolve()
    if not str(dest).startswith(str(base)):
        raise ValueError("invalid path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    md5 = dvc_io.dvc_add(rel.as_posix())
    dvc_io.dvc_push()

    row = DatasetVersion(
        dataset_id=ds.id,
        version=next_v,
        dvc_md5=md5,
        storage_key=dvc_io.dvc_remote_key(md5),
        original_filename=filename,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        target_column=target_column,
        column_stats=column_stats(df),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_versions(db: Session, dataset_id: int) -> list[DatasetVersion]:
    return (
        db.query(DatasetVersion)
        .filter_by(dataset_id=dataset_id)
        .order_by(DatasetVersion.version)
        .all()
    )


def get_version(db: Session, dataset_id: int, version: int) -> DatasetVersion | None:
    return db.query(DatasetVersion).filter_by(dataset_id=dataset_id, version=version).first()


def version_content(row: DatasetVersion) -> bytes:
    return dvc_io.read_content(row.dvc_md5)
