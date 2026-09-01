import io
import subprocess
from pathlib import Path

import boto3
import yaml
from botocore.client import Config

from ..core.config import settings

REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {(r.stderr or r.stdout).strip()[:500]}")
    return r.stdout


def dvc_add(rel_path: str) -> str:
    """dvc add a repo-relative file, return the content md5."""
    _run(["dvc", "add", rel_path])
    meta = yaml.safe_load((REPO / f"{rel_path}.dvc").read_text())
    for out in meta.get("outs", []):
        # .dvc stores path relative to the .dvc file's own directory
        if Path(out.get("path", "")).name == Path(rel_path).name:
            return out["md5"]
    raise RuntimeError(f"no md5 found in {rel_path}.dvc")


def dvc_push() -> None:
    _run(["dvc", "push"])


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def dvc_remote_key(md5: str) -> str:
    # DVC's deterministic s3 remote layout
    return f"files/md5/{md5[:2]}/{md5[2:]}"


def fetch_remote_content(md5: str) -> bytes:
    """Pull the object back out of MinIO using the DVC md5 layout."""
    return s3_client().get_object(Bucket=settings.bucket_dvc, Key=dvc_remote_key(md5))["Body"].read()


def cache_path_for(md5: str) -> Path:
    return REPO / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]


def read_content(md5: str) -> bytes:
    """Remote-first read, local dvc cache fallback."""
    try:
        return fetch_remote_content(md5)
    except Exception:
        p = cache_path_for(md5)
        if p.exists():
            return p.read_bytes()
        raise


def put_artifact(key: str, data: bytes) -> str:
    """Drop a pipeline artifact (parquet/report/plot) into the artifacts bucket."""
    s3_client().put_object(Bucket=settings.bucket_artifacts, Key=key, Body=data)
    return f"s3://{settings.bucket_artifacts}/{key}"


def get_artifact(key: str) -> bytes:
    return s3_client().get_object(Bucket=settings.bucket_artifacts, Key=key)["Body"].read()
