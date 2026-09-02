"""Step 4: train — Optuna HPO (nested MLflow runs) -> final model -> register house-price-sk."""
import io
import json
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from common import TARGET, WORKFLOW_NAME, clean_features, get_artifact, put_json
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ID_COL = "Id"


def load_split():
    features = pd.read_parquet(io.BytesIO(get_artifact(f"houses/{WORKFLOW_NAME}/features.parquet")))
    X = features.drop(columns=[c for c in (ID_COL, TARGET) if c in features.columns])
    y = features[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(params):
    family = params["family"]
    if family == "histgb":
        return HistGradientBoostingRegressor(
            learning_rate=params["lr"],
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=params["l2"],
            random_state=42,
        )
    if family == "rf":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=2,
        )
    raise ValueError(family)


def sample_params(trial: optuna.Trial) -> dict:
    family = trial.suggest_categorical("family", ["histgb", "rf"])
    params = {"family": family}
    if family == "histgb":
        params["lr"] = trial.suggest_float("lr", 0.02, 0.3, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 200, 600, step=100)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 8, 48, step=8)
        params["l2"] = trial.suggest_float("l2", 1e-3, 10.0, log=True)
    else:
        params["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=100)
        params["max_depth"] = trial.suggest_int("max_depth", 6, 24)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
    return params


def main() -> None:
    idx = sys.argv.index("--n_trials") + 1 if "--n_trials" in sys.argv else None
    n_trials = int(sys.argv[idx]) if idx else 15

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("house-prices")

    X_tr, X_te, y_tr, y_te = load_split()
    X_tr, X_te = clean_features(X_tr, X_te)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
            model = build_model(params).fit(X_tr, y_tr)
            pred = model.predict(X_te)
            rmse = float(np.sqrt(((y_te - pred) ** 2).mean()))
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": float(mean_absolute_error(y_te, pred)), "r2": float(r2_score(y_te, pred))})
            mlflow.set_tag("optuna_trial", str(trial.number))
            trial.set_user_attr("run_id", run.info.run_id)
        return rmse

    # Use context manager so parent run is closed even if optimize fails
    with mlflow.start_run(run_name="optuna-study") as parent:
        study.optimize(objective, n_trials=n_trials)
        if not study.best_trial or len([t for t in study.trials if t.state.name == "COMPLETE"]) == 0:
            raise SystemExit("no successful trials")
        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_trial_rmse", study.best_value)
        parent_run_id = parent.info.run_id

    # final model with the winning hyperparameters, logged + registered
    with mlflow.start_run(run_name="final-model") as final:
        model = build_model(best.params).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics = {
            "rmse": float(np.sqrt(((y_te - pred) ** 2).mean())),
            "mae": float(mean_absolute_error(y_te, pred)),
            "r2": float(r2_score(y_te, pred)),
        }
        mlflow.log_params(best.params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X_tr.head(50), model.predict(X_tr.head(50))),
            registered_model_name="house-price-sk",
        )
        final_run_id = final.info.run_id

    # the registered version that log_model just created for this run
    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    versions = sorted(c.search_model_versions(f"name='house-price-sk' and run_id='{final_run_id}'"), key=lambda m: int(m.version))
    version = int(versions[-1].version)

    result = {
        "model_name": "house-price-sk",
        "model_version": version,
        "run_id": final_run_id,
        "study_run_id": parent_run_id,
        "best_params": json.loads(json.dumps(best.params)),
        "n_trials": n_trials,
        "metrics": metrics,
    }
    put_json(f"houses/{WORKFLOW_NAME}/train_result.json", result)
    print("registered house-price-sk v", version, "rmse=", round(metrics["rmse"], 2), sep="")


if __name__ == "__main__":
    main()
