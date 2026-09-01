import os

import mlflow
from mlflow.tracking import MlflowClient

from ..core.config import settings


def client() -> MlflowClient:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"http://{settings.minio_endpoint}")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def list_models() -> list[dict]:
    c = client()
    out = []
    for rm in c.search_registered_models():
        latest = {v.current_stage: v.version for v in (rm.latest_versions or [])}
        out.append(
            {
                "name": rm.name,
                "description": rm.description,
                "latest": latest,
                "n_versions": sum(1 for _ in c.search_model_versions(f"name='{rm.name}'")),
            }
        )
    return out


def list_versions(name: str) -> list[dict]:
    c = client()
    return [
        {
            "version": int(mv.version),
            "stage": mv.current_stage,
            "status": mv.status,
            "run_id": mv.run_id,
            "source": mv.source,
            "created_at": mv.creation_timestamp,
        }
        for mv in sorted(
            c.search_model_versions(f"name='{name}'"),
            key=lambda m: int(m.version),
            reverse=True,
        )
    ]


def get_model_version(name: str, version: int | str):
    return client().get_model_version(name, str(version))


def production_version(name: str) -> dict | None:
    """Current Production-stage version, or None."""
    for mv in client().search_model_versions(f"name='{name}'"):
        if mv.current_stage.lower() == settings.prod_stage.lower():
            return {"name": name, "version": int(mv.version), "run_id": mv.run_id}
    return None


def promote(name: str, version: int | str) -> dict:
    c = client()
    mv = c.get_model_version(name, str(version))  # raises if it doesn't exist
    c.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=settings.prod_stage,
        archive_existing_versions=True,
    )
    return {"name": name, "version": int(mv.version), "stage": settings.prod_stage}
