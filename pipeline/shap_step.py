"""Step 7: shap — TreeExplainer summary for the winning sklearn model, logged to its run."""
import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.sklearn
import pandas as pd
import shap
from common import TARGET, WORKFLOW_NAME, get_artifact, get_json


def main() -> None:
    train = get_json(f"houses/{WORKFLOW_NAME}/train_result.json")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    features = pd.read_parquet(io.BytesIO(get_artifact(f"houses/{WORKFLOW_NAME}/features.parquet")))
    X = features.drop(columns=[c for c in ("Id", TARGET) if c in features.columns])

    model = mlflow.sklearn.load_model(f"runs:/{train['run_id']}/model")
    explainer = shap.TreeExplainer(model)
    sample = X.sample(min(200, len(X)), random_state=42)
    sv = explainer.shap_values(sample)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plot_path = os.path.join(tempfile.gettempdir(), "shap_summary.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    import numpy as np

    mean_abs = np.abs(sv).mean(axis=0)
    top = sorted(zip(X.columns, mean_abs), key=lambda t: -t[1])[:20]
    top_json = [{"feature": f, "mean_abs_shap": float(v)} for f, v in top]

    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    c.log_artifact(train["run_id"], plot_path, artifact_path="shap")

    import json

    top_path = os.path.join(tempfile.gettempdir(), "shap_top20.json")
    with open(top_path, "w") as f:
        json.dump({"model": train["model_name"], "version": train["model_version"], "top_features": top_json}, f, indent=2)
    c.log_artifact(train["run_id"], top_path, artifact_path="shap")

    # also park in the artifact bucket so the model-card endpoint can render without MLflow round-trip
    with open(top_path, "rb") as f:
        from common import put_artifact

        put_artifact(f"houses/{WORKFLOW_NAME}/shap_top20.json", f.read())
    print("shap artifacts logged to run", train["run_id"], "- top feature:", top_json[0]["feature"])


if __name__ == "__main__":
    main()
