"""Step 5: train-nn — small PyTorch MLP on the same split, registered as house-price-nn."""
import io
import os

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from common import TARGET, WORKFLOW_NAME, clean_features, get_artifact, put_json
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ID_COL = "Id"
EPOCHS = 300


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    torch.manual_seed(42)
    features = pd.read_parquet(io.BytesIO(get_artifact(f"houses/{WORKFLOW_NAME}/features.parquet")))
    X = features.drop(columns=[c for c in (ID_COL, TARGET) if c in features.columns])
    y = features[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = clean_features(X_tr, X_te)

    scaler = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
    ytr = torch.tensor(y_tr.values, dtype=torch.float32)
    yte_t = torch.tensor(y_te.values, dtype=torch.float32)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("house-prices")

    with mlflow.start_run(run_name="mlp-final") as run:
        model = MLP(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()
            if (epoch + 1) % 100 == 0:
                print(f"epoch {epoch + 1}: train mse {loss.item():.0f}")

        model.eval()
        with torch.no_grad():
            pred = model(Xte).numpy()
        y_pred = np.asarray(pred, dtype=float)
        metrics = {
            "rmse": float(np.sqrt(((y_te.values - y_pred) ** 2).mean())),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "r2": float(r2_score(y_te, y_pred)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({"framework": "pytorch", "epochs": EPOCHS, "n_features": X.shape[1]})

        class Wrapper(torch.nn.Module):
            def __init__(self, m, scaler):
                super().__init__()
                self.m = m
                self.scaler = scaler

            def forward(self, x):
                return self.m(x).squeeze(-1)

        # log scaler + model: the scaler is part of the pipeline contract, saved alongside
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="house-price-nn",
            pip_requirements=["torch", "numpy"],
        )

        from mlflow.tracking import MlflowClient

        c = MlflowClient()
        c.log_artifact(run.info.run_id, _scaler_to_file(scaler), "model")

        versions = sorted(c.search_model_versions(f"name='house-price-nn' and run_id='{run.info.run_id}'"), key=lambda m: int(m.version))
        version = int(versions[-1].version)

    result = {
        "model_name": "house-price-nn",
        "model_version": version,
        "run_id": run.info.run_id,
        "metrics": metrics,
    }
    put_json(f"houses/{WORKFLOW_NAME}/nn_result.json", result)
    print("registered house-price-nn v", version, "rmse=", round(metrics["rmse"], 2), sep="")


def _scaler_to_file(scaler: StandardScaler) -> str:
    import pickle
    import tempfile

    p = os.path.join(tempfile.gettempdir(), "scaler.pkl")
    with open(p, "wb") as f:
        pickle.dump(scaler, f)
    return p


if __name__ == "__main__":
    main()
