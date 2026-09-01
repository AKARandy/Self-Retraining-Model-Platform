"""Step 6: evaluate — compare the freshly trained model against current Production; write decision."""
import io
import os

import mlflow.sklearn
import numpy as np
import pandas as pd

from common import TARGET, WORKFLOW_NAME, get_artifact, get_json, put_json


def main() -> None:
    train = get_json(f"houses/{WORKFLOW_NAME}/train_result.json")
    try:
        nn = get_json(f"houses/{WORKFLOW_NAME}/nn_result.json")
    except Exception:
        nn = None

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    from mlflow.tracking import MlflowClient

    c = MlflowClient()

    # current production model (if any) and its rmse logged at its own train time
    prod = None
    for mv in c.search_model_versions("name='house-price-sk'"):
        if mv.current_stage.lower() == "production":
            prod = mv
            break

    new_rmse = train["metrics"]["rmse"]
    if prod is None:
        promote, reason = True, "no production model yet"
    else:
        prod_metrics = c.get_run(prod.run_id).data.metrics
        prod_rmse = prod_metrics.get("rmse")
        if prod_rmse is None:
            promote, reason = True, "production model has no logged rmse"
        else:
            improve = (prod_rmse - new_rmse) / prod_rmse
            promote = improve > 0.0
            reason = f"new rmse {new_rmse:.1f} vs prod {prod_rmse:.1f} ({improve * 100:+.2f}%)"

    decision = {
        "promote": promote,
        "reason": reason,
        "candidate": {"name": train["model_name"], "version": train["model_version"], "metrics": train["metrics"]},
        "candidate_nn": nn,
        "production": {"name": prod.name, "version": int(prod.version), "run_id": prod.run_id} if prod else None,
    }
    put_json(f"houses/{WORKFLOW_NAME}/decision.json", decision)
    print("promote:", promote, "-", reason)


if __name__ == "__main__":
    main()
