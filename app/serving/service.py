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
_scaler_cache: dict = {}  # (name, version, run_id) -> scaler


def resolve_target(model_name: str | None) -> dict:
    """model='auto' -> production house-price-sk. Named models -> production if
    promoted, else their latest registered version."""
    if not model_name or model_name == "auto":
        name = settings.default_model
    else:
        name = model_name
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
    key = (name, version, run_id)
    if key in _scaler_cache:
        return _scaler_cache[key]
    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    p = c.download_artifacts(run_id, "model/scaler.pkl", tempfile.gettempdir())
    # Simple RCE mitigation: restrict unpickling to known safe types
    import sklearn.preprocessing  # noqa: F401 — allow

    class _SafeUnpickler(pickle.Unpickler):
        _allow = {
            ("sklearn.preprocessing._data", "StandardScaler"),
            ("sklearn.preprocessing._data", "MinMaxScaler"),
            ("sklearn.preprocessing._data", "RobustScaler"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy", "dtype"),
            ("numpy", "ndarray"),
            ("builtins", "dict"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "set"),
        }

        def find_class(self, module, name):
            if (module, name) in self._allow or module.startswith("sklearn.") or module.startswith("numpy."):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked pickle class {module}.{name}")

    with open(p, "rb") as f:
        scaler = _SafeUnpickler(f).load()
    _scaler_cache[key] = scaler
    return scaler


def predict(features: dict, model_name: str | None = None) -> tuple[float, dict, dict]:
    """Returns (prediction, model_key, engineered_features)."""
    from . import featurizer

    recipe = featurizer.load_recipe()
    engineered = featurizer.featurize(features, recipe)
    # Resolve once and reuse for both model and scaler (avoids double MLflow lookup).
    target = resolve_target(model_name)
    model, key = get_model(model_name)
    # get_model re-resolves; ensure key matches target — if mismatch (race), trust target's run_id.

    if key[0] == "house-price-nn":
        import torch

        cols = list(recipe["feature_columns"])
        vec = featurizer.align(cols, engineered, recipe)
        X = pd.DataFrame([vec])[cols]
        scaler = _nn_scaler(target["name"], target["version"], target["run_id"])
        xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = float(model(xs).item())
        return pred, key, engineered

    cols = list(getattr(model, "feature_names_in_", recipe["feature_columns"]))
    X = pd.DataFrame([featurizer.align(cols, engineered, recipe)])[cols]
    pred = float(model.predict(X)[0])
    return pred, key, engineered
