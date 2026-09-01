from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.security import require_api_key
from ..core.db import get_db
from ..core.models import DriftCheck
from . import service

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/check-drift", dependencies=[Depends(require_api_key)])
def check(db: Session = Depends(get_db), dataset_version_id: int = Query(1), min_window: int = Query(20)):
    check = service.check_drift(db, dataset_version_id, min_window)
    return {
        "check_id": check.id,
        "verdict": check.verdict,
        "triggered_retrain": check.triggered_retrain,
        "training_run_id": check.training_run_id,
        "features_drifted": check.features_drifted,
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
