#!/bin/sh
set -e

# .dvc/config is baked with host-friendly values; rewrite the remote for in-network access.
# Credentials come from compose env (minioadmin pair), never baked into the image.
dvc remote modify minio endpointurl "$MINIO_ENDPOINT" --local
dvc remote modify minio access_key_id "$AWS_ACCESS_KEY_ID" --local
dvc remote modify minio secret_access_key "$AWS_SECRET_ACCESS_KEY" --local

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
