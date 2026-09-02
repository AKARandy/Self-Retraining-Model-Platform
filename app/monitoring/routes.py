from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.audit import record as audit_record
from ..core.db import get_db
from ..core.models import DriftCheck
from ..core.security import require_api_key
from . import service

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/check-drift", dependencies=[Depends(require_api_key)])
def check(db: Session = Depends(get_db), dataset_version_id: int = Query(1), min_window: int = Query(20)):
    chk = service.check_drift(db, dataset_version_id, min_window)
    # Contract: every write attempt is audited — successful drift checks too.
    # check_drift already committed its DriftCheck; this audit is a second txn with rollback safety
    try:
        audit_record(
            db,
            action="POST /monitoring/check-drift",
            resource=f"dataset_version/{dataset_version_id}",
            allowed=True,
            detail={"verdict": chk.verdict, "triggered_retrain": chk.triggered_retrain, "training_run_id": chk.training_run_id},
        )
    except Exception:
        # Audit failure must not hide the drift result
        import logging

        logging.getLogger(__name__).exception("audit failed for check-drift")
    return {
        "check_id": chk.id,
        "verdict": chk.verdict,
        "triggered_retrain": chk.triggered_retrain,
        "training_run_id": chk.training_run_id,
        "features_drifted": chk.features_drifted,
    }


@router.get("/drift-checks")
def history(db: Session = Depends(get_db)):
    rows = db.query(DriftCheck).order_by(DriftCheck.id.desc()).limit(20).all()
    return [
        {
            "id": r.id,
            "verdict": r.verdict,
            "triggered_retrain": r.triggered_retrain,
            "training_run_id": r.training_run_id,
            "created_at": str(r.created_at),
        }
        for r in rows
    ]
