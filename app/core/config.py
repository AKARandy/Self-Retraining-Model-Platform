from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "mlops-monolith"
    api_key: str = "dev-key"
    database_url: str = "postgresql+psycopg://mlops:mlops@localhost:5433/mlops"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    bucket_dvc: str = "dvc-store"
    bucket_artifacts: str = "pipeline-artifacts"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    argo_url: str = "https://localhost:2746"
    argo_token: str = ""
    argo_namespace: str = "argo"
    workflow_template: str = "train-pipeline"

    # how containers inside minikube reach services on the Windows host
    host_svc: str = "host.minikube.internal"

    # serving
    prod_stage: str = "Production"
    default_model: str = "house-price-sk"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
