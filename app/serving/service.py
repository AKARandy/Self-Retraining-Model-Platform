import os
import pickle
import tempfile

import mlflow.pytorch
import mlflow.sklearn
import pandas as pd

from ..core.config import settings
from ..registry import service as registry


def _mlflow_env():
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"http://{settings.minio_endpoint}")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


_prod_cache: dict = {"key": None, "model": None}


def resolve_target(model_name: str | None) -> dict:
    """model='auto' -> production house-price-sk. Named models -> production if
    promoted, else their latest registered version."""
    name = model_name or settings.default_model
    prod = registry.production_version(name)
    if prod:
        return prod
    versions = registry.list_versions(name)
    if not versions:
        raise LookupError(f"model {name!r} has no registered versions")
    return {"name": name, "version": versions[0]["version"], "run_id": versions[0]["run_id"]}


def get_model(name: str | None):
    target = resolve_target(name)
    key = (target["name"], target["version"])
    if _prod_cache["key"] != key:
        _mlflow_env()
        if target["name"] == "house-price-nn":
            model = mlflow.pytorch.load_model(f"models:/{target['name']}/{target['version']}")
        else:
            model = mlflow.sklearn.load_model(f"models:/{target['name']}/{target['version']}")
        _prod_cache["key"] = key
        _prod_cache["model"] = model
    return _prod_cache["model"], key


def _nn_scaler(name: str, version: int, run_id: str):
    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    p = c.download_artifacts(run_id, "model/scaler.pkl", tempfile.gettempdir())
    with open(p, "rb") as f:
        return pickle.load(f)


def predict(features: dict, model_name: str | None = None) -> tuple[float, dict, dict]:
    """Returns (prediction, model_key, engineered_features)."""
    from . import featurizer

    recipe = featurizer.load_recipe()
    engineered = featurizer.featurize(features, recipe)
    model, key = get_model(model_name)

    if key[0] == "house-price-nn":
        import numpy as np
        import torch

        cols = list(recipe["feature_columns"])
        vec = featurizer.align(cols, engineered, recipe)
        X = pd.DataFrame([vec])[cols]
        scaler = _nn_scaler(key[0], key[1], resolve_target(model_name)["run_id"])
        xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = float(model(xs).item())
        return pred, key, engineered

    cols = list(getattr(model, "feature_names_in_", recipe["feature_columns"]))
    X = pd.DataFrame([featurizer.align(cols, engineered, recipe)])[cols]
    pred = float(model.predict(X)[0])
    return pred, key, engineered
