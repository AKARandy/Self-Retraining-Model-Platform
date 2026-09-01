"""Smoke test: log a real sklearn run + artifact to MLflow (backend=Postgres, artifacts=MinIO),
register it as house-price-sk — promoted later through the API endpoint."""
import json
import os

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402

os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{settings.minio_endpoint}"
os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("smoke")

rng = np.random.default_rng(7)
X = rng.normal(size=(100, 3))
y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(scale=0.1, size=100)

with mlflow.start_run(run_name="smoke-linear") as run:
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    mlflow.log_metric("r2", float(model.score(X, y)))
    mlflow.log_param("framework", "sklearn")
    # log a small JSON artifact so the artifact store round-trip is proven
    mlflow.log_dict({"coef": model.coef_.tolist(), "intercept": model.intercept_}, "smoke_info.json")
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(X, preds),
        registered_model_name="house-price-sk",
    )
    print("RUN_ID:", run.info.run_id)
    print("METRICS:", json.dumps({k: v for k, v in run.data.metrics.items()}))
