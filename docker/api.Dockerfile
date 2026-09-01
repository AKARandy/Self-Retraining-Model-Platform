# API container — the FastAPI monolith + its DVC working copy.
# Kept separate from docker/requirements-training.txt (the cluster image) on purpose:
# the API needs mlflow/sklearn/torch for serving, not featuretools.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .dvc ./.dvc
COPY docker/api-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p data/raw

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# entrypoint repoints the DVC remote at the in-network MinIO before serving
ENTRYPOINT ["/entrypoint.sh"]
