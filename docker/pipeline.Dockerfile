FROM python:3.12-slim

WORKDIR /work

COPY docker/requirements-training.txt ./requirements-training.txt
RUN pip install --no-cache-dir -r requirements-training.txt

COPY pipeline/ /work/pipeline/

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_S3_ENDPOINT_URL=http://host.minikube.internal:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    AWS_DEFAULT_REGION=us-east-1

# each WorkflowTemplate step runs: python pipeline/<step>.py
CMD ["python", "pipeline/ingest.py"]
