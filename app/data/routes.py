from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..core.audit import record
from ..core.db import get_db
from ..core.models import Dataset
from ..core.security import require_api_key
from . import service

router = APIRouter(prefix="/datasets", tags=["data"])


@router.post("", dependencies=[Depends(require_api_key)])
def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    target_column: str = Form(""),
    db: Session = Depends(get_db),
):
    content = file.file.read()
    if not content:
        raise HTTPException(400, "empty file")
    try:
        row = service.upload_version(
            db,
            name=name or (file.filename or "dataset").rsplit(".", 1)[0],
            filename=file.filename or "upload.csv",
            content=content,
            target_column=target_column or None,
        )
    except Exception as e:
        raise HTTPException(500, f"upload failed: {e}")
    record(db, "upload_dataset", f"{row.dataset_id}/v{row.version}", allowed=True, detail={"dvc_md5": row.dvc_md5})
    return {
        "dataset_id": row.dataset_id,
        "version": row.version,
        "dvc_md5": row.dvc_md5,
        "n_rows": row.n_rows,
        "n_cols": row.n_cols,
        "storage_key": row.storage_key,
    }


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    return [{"id": d.id, "name": d.name, "created_at": d.created_at} for d in db.query(Dataset).all()]


@router.get("/{dataset_id}/versions")
def get_versions(dataset_id: int, db: Session = Depends(get_db)):
    rows = service.list_versions(db, dataset_id)
    if not rows:
        raise HTTPException(404, f"no versions for dataset {dataset_id}")
    return [
        {
            "version": r.version,
            "dvc_md5": r.dvc_md5,
            "n_rows": r.n_rows,
            "n_cols": r.n_cols,
            "target_column": r.target_column,
            "original_filename": r.original_filename,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@router.get("/{dataset_id}/versions/{version}/content")
def get_content(dataset_id: int, version: int, db: Session = Depends(get_db)):
    row = service.get_version(db, dataset_id, version)
    if row is None:
        raise HTTPException(404, f"dataset {dataset_id} v{version} not found")
    data = service.version_content(row)
    return Response(content=data, media_type="text/csv", headers={"X-DVC-MD5": row.dvc_md5})
