"""Shared config + IO for pipeline step containers.

Everything reaches host services through host.minikube.internal:
API :8000, MLflow :5000, MinIO :9000, Postgres :5433.
Artifacts flow between steps through the MinIO `pipeline-artifacts` bucket.
"""
import hashlib
import os
from functools import lru_cache

import boto3
from botocore.client import Config

ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET", "pipeline-artifacts")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "dvc-store")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://host.minikube.internal:9000")
API_URL = os.getenv("API_URL", "http://host.minikube.internal:8000")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://mlops:mlops@host.minikube.internal:5433/mlops",
)
WORKFLOW_NAME = os.getenv("WORKFLOW_NAME", "local")
TARGET = os.getenv("TARGET", "SalePrice")

MINIO_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@lru_cache
def s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def put_artifact(key: str, data: bytes) -> str:
    s3().put_object(Bucket=ARTIFACT_BUCKET, Key=key, Body=data)
    return key


def get_artifact(key: str) -> bytes:
    return s3().get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"].read()


def put_json(key: str, obj) -> str:
    import json

    return put_artifact(key, json.dumps(obj, indent=2, default=str).encode())


def get_json(key: str):
    import json

    return json.loads(get_artifact(key))


@lru_cache
def db_engine():
    from sqlalchemy import create_engine

    return create_engine(DB_URL, pool_pre_ping=True)


def fetch_dataset_content(dataset_id: int, dataset_version: int) -> bytes:
    """Pull the CSV through the API (which reads it from the DVC/MinIO remote)."""
    import requests

    url = f"{API_URL}/datasets/{dataset_id}/versions/{dataset_version}/content"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def clean_features(X_tr, X_te):
    """Replace inf, drop all-NaN columns, median-fill the rest (medians from train only)."""
    import numpy as np

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)
    all_nan = X_tr.columns[X_tr.isna().all()]
    X_tr = X_tr.drop(columns=list(all_nan))
    X_te = X_te.drop(columns=list(all_nan))
    med = X_tr.median()
    return X_tr.fillna(med), X_te.fillna(med)
