
import httpx
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.db import SessionLocal
from ..core.models import DatasetVersion, TrainingRun


def _argo_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if settings.argo_token:
        h["Authorization"] = f"Bearer {settings.argo_token}"
    return h


def submit_run(
    db: Session,
    dataset_id: int,
    dataset_version: int | None = None,
    n_trials: int = 15,
) -> TrainingRun:
    dvq = db.query(DatasetVersion).filter_by(dataset_id=dataset_id)
    if dataset_version is None:
        dv = dvq.order_by(DatasetVersion.version.desc()).first()
    else:
        dv = dvq.filter_by(version=dataset_version).first()
    if dv is None:
        raise ValueError(f"dataset {dataset_id} v{dataset_version} not registered")

    body = {
        "namespace": settings.argo_namespace,
        "resourceKind": "WorkflowTemplate",
        "resourceName": settings.workflow_template,
        "submitOptions": {
            # SubmitOptions.parameters is a list of "name=value" strings (CLI-compatible)
            "parameters": [
                f"dataset_id={dataset_id}",
                f"dataset_version={dv.version}",
                f"dataset_version_id={dv.id}",
                f"n_trials={n_trials}",
            ]
        },
    }
    r = httpx.post(
        f"{settings.argo_url}/api/v1/workflows/{settings.argo_namespace}/submit",
        json=body,
        headers=_argo_headers(),
        verify=False,
        timeout=60,
    )
    r.raise_for_status()
    wf = r.json()
    meta = wf["metadata"]

    run = TrainingRun(
        argo_uid=meta.get("uid"),
        argo_name=meta.get("name"),
        status="submitted",
        params={"dataset_id": dataset_id, "dataset_version": dv.version, "n_trials": n_trials},
        dataset_version_id=dv.id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def sync_status(run: TrainingRun) -> TrainingRun:
    """Pull the workflow's phase from Argo into the local training_runs row."""
    if not run.argo_name:
        return run
    r = httpx.get(
        f"{settings.argo_url}/api/v1/workflows/{settings.argo_namespace}/{run.argo_name}",
        headers=_argo_headers(),
        verify=False,
        timeout=30,
    )
    if r.status_code != 200:
        return run
    wf = r.json()
    phase = (wf.get("status") or {}).get("phase", "Unknown")

    changed = False
    if phase != run.status:
        run.status = phase
        changed = True
    if phase in ("Succeeded", "Failed", "Error") and run.finished_at is None:
        from datetime import datetime, timezone

        run.finished_at = datetime.now(timezone.utc)
        if phase != "Succeeded":
            nodes = (wf.get("status") or {}).get("nodes") or {}
            errors = [
                n.get("message")
                for n in nodes.values()
                if n.get("phase") in ("Failed", "Error") and n.get("message")
            ]
            run.error = "; ".join(errors)[:2000] or None
        changed = True
    if changed:
        db = SessionLocal()
        try:
            db.merge(run)
            db.commit()
        finally:
            db.close()
    return run


def run_payload(run: TrainingRun) -> dict:
    return {
        "id": run.id,
        "status": run.status,
        "argo_name": run.argo_name,
        "params": run.params,
        "dataset_version_id": run.dataset_version_id,
        "error": run.error,
        "created_at": str(run.created_at),
        "finished_at": str(run.finished_at) if run.finished_at else None,
        "argo_ui": f"{settings.argo_url}/workflows/{settings.argo_namespace}/{run.argo_name}" if run.argo_name else None,
    }
