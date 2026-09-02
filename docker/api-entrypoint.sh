#!/bin/sh
set -e

# Alembic owns the app `mlops` schema — migrate before serving (idempotent).
# Uses DATABASE_URL from compose env; never touches MLflow's `mlflow` DB.
if [ -f ./alembic.ini ]; then
  echo "migrating DB (alembic upgrade head)..."
  alembic upgrade head || { echo "alembic upgrade failed"; exit 1; }
else
  echo "alembic.ini not found — skipping migration (dev mode)"
fi

# .dvc/config is baked with host-friendly values; rewrite the remote for in-network access.
# Credentials come from compose env (minioadmin pair), never baked into the image.
# Fix scheme: compose provides minio:9000 (no http://), but DVC needs http://host:port
if [ -n "${MINIO_ENDPOINT:-}" ]; then
  _ep="$MINIO_ENDPOINT"
  case "$_ep" in http://*|https://*) ;; *) _ep="http://$_ep" ;; esac
  dvc remote modify minio endpointurl "$_ep" --local || true
fi
if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  dvc remote modify minio access_key_id "$AWS_ACCESS_KEY_ID" --local || true
fi
if [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  dvc remote modify minio secret_access_key "$AWS_SECRET_ACCESS_KEY" --local || true
fi

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
