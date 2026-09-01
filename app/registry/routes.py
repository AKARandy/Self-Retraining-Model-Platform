from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from mlflow.exceptions import MlflowException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.audit import record
from ..core.config import settings
from ..core.db import get_db
from ..core.security import require_api_key
from . import service
from .card import render_card

router = APIRouter(prefix="/registry", tags=["registry"])


class PromoteBody(BaseModel):
    name: str
    version: int


@router.get("/models")
def models():
    return service.list_models()


@router.get("/models/{name}/versions")
def versions(name: str):
    try:
        return service.list_versions(name)
    except MlflowException as e:
        raise HTTPException(404, str(e))


@router.get("/models/{name}/current")
def current(name: str):
    prod = service.production_version(name)
    if prod is None:
        raise HTTPException(404, f"no Production version for {name}")
    return prod


@router.get("/models/{name}/versions/{version}/card", response_class=PlainTextResponse)
def model_card(name: str, version: int):
    try:
        mv = service.get_model_version(name, version)
        run = service.client().get_run(mv.run_id)
    except MlflowException as e:
        raise HTTPException(404, str(e))
    return render_card(mv, run, settings.mlflow_tracking_uri)


@router.post("/promote", dependencies=[Depends(require_api_key)])
def promote(body: PromoteBody, db: Session = Depends(get_db)):
    try:
        result = service.promote(body.name, body.version)
    except MlflowException as e:
        record(db, "promote", f"{body.name}/{body.version}", allowed=False, detail={"error": str(e)[:300]})
        raise HTTPException(404, str(e))
    record(db, "promote", f"{body.name}/{body.version}", allowed=True, detail=result)
    return result
