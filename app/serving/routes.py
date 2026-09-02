from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.audit import record as audit_record
from ..core.db import get_db
from ..core.models import Prediction
from ..core.security import require_api_key
from . import service

router = APIRouter(tags=["serving"])


class PredictBody(BaseModel):
    features: dict
    model: str | None = None  # auto | house-price-sk | house-price-nn


@router.post("/predict", dependencies=[Depends(require_api_key)])
def predict(body: PredictBody, db: Session = Depends(get_db)):
    try:
        pred, key, engineered = service.predict(body.features, body.model)
    except LookupError as e:
        raise HTTPException(404, str(e))
    row = Prediction(
        model_name=key[0],
        model_version=key[1],
        source="live",
        features=body.features,
        prediction=pred,
    )
    # Atomic: prediction + audit in one transaction (simple fix)
    db.add(row)
    audit_record(db, action="POST /predict", resource=f"{key[0]}/{key[1]}", allowed=True, detail={"n_engineered": len(engineered)}, commit=False)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"prediction": pred, "model": key[0], "model_version": key[1], "n_engineered_features": len(engineered)}


@router.get("/serving/current")
def current(model: str | None = None):
    try:
        return service.resolve_target(model)
    except LookupError as e:
        raise HTTPException(404, str(e))
