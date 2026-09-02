# API container — the FastAPI monolith + its DVC working copy.
# Kept separate from docker/requirements-training.txt (the cluster image) on purpose:
# the API needs mlflow/sklearn/torch for serving, not featuretools.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY .dvc ./.dvc
COPY docker/api-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p data/raw

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# entrypoint migrates DB (alembic upgrade head) and repoints DVC remote before serving
ENTRYPOINT ["/entrypoint.sh"]
