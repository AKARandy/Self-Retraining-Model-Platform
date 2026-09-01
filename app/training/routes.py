from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.audit import record
from ..core.db import get_db
from ..core.models import TrainingRun
from ..core.security import require_api_key
from . import service

router = APIRouter(prefix="/train-runs", tags=["training"])


class SubmitBody(BaseModel):
    dataset_id: int = 1
    dataset_version: int | None = None
    n_trials: int = 15


@router.post("", dependencies=[Depends(require_api_key)])
def submit(body: SubmitBody, db: Session = Depends(get_db)):
    try:
        run = service.submit_run(db, body.dataset_id, body.dataset_version, body.n_trials)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"argo submit failed: {e}")
    record(db, "submit_training", run.argo_name or f"run/{run.id}", allowed=True, detail=body.model_dump())
    return service.run_payload(run)


@router.get("")
def list_runs(db: Session = Depends(get_db)):
    rows = db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(50).all()
    return [service.run_payload(r) for r in rows]


@router.get("/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter_by(id=run_id).first()
    if run is None:
        raise HTTPException(404, f"training run {run_id} not found")
    return service.run_payload(service.sync_status(run))
