"""Batch scoring job (runs inside the mlops-pipeline image via Argo CronWorkflow):
score the latest feature set head through the current production model and
write predictions rows (source=batch) straight into Postgres."""
import io
import os

import pandas as pd
from sqlalchemy import create_engine, text

from common import DB_URL, WORKFLOW_NAME, get_artifact, get_json

BATCH_SIZE = 100


def main() -> None:
    recipe_pointer = get_json("houses/recipe/latest.json")
    features_bytes = get_artifact(f"houses/{recipe_pointer['workflow']}/features.parquet")
    features = pd.read_parquet(io.BytesIO(features_bytes))

    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    prod = None
    for mv in c.search_model_versions("name='house-price-sk'"):
        if mv.current_stage.lower() == "production":
            prod = mv
            break
    if prod is None:
        print("no production model — nothing to score")
        return

    model = mlflow.sklearn.load_model(f"models:/{prod.name}/{prod.version}")
    cols = list(getattr(model, "feature_names_in_", []))
    # align: model may pre/post-date the latest feature parquet
    X = features.reindex(columns=cols, fill_value=0.0).head(BATCH_SIZE) if cols else features.head(BATCH_SIZE)

    preds = model.predict(X)
    engine = create_engine(DB_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        for p in preds:
            conn.execute(
                text(
                    "INSERT INTO predictions (model_name, model_version, source, features, prediction)"
                    " VALUES (:m, :v, 'batch', CAST(NULL AS JSONB), :p)"
                ),
                {"m": prod.name, "v": int(prod.version), "p": float(p)},
            )
    print(f"batch: scored {len(preds)} rows with {prod.name} v{prod.version}")


if __name__ == "__main__":
    main()
