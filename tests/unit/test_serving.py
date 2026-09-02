"""Recipe loading, transforms, alignment, serving success/failure, prediction rows."""
from unittest.mock import patch

import pytest


def test_featurizer_and_align():
    from app.serving import featurizer

    recipe = {
        "passthrough_numeric": ["LotArea"],
        "onehot": {"Neighborhood": ["CollgCr", "OldTown"]},
        "transforms": [{"column": "LotArea_GrLivArea_add", "op": "add", "a": "LotArea", "b": "GrLivArea"}],
        "aggs": [],
        "agg_values": {},
        "medians": {"LotArea": 5000},
        "feature_columns": ["LotArea", "Neighborhood_CollgCr", "Neighborhood_OldTown", "Neighborhood_nan", "LotArea_GrLivArea_add"],
    }
    engineered = featurizer.featurize({"LotArea": 1000, "GrLivArea": 500, "Neighborhood": "CollgCr"}, recipe)
    assert abs(engineered["LotArea"] - 1000) < 1e-6
    assert engineered["Neighborhood_CollgCr"] == 1.0
    assert engineered["LotArea_GrLivArea_add"] == 1500
    # align with medians fallback
    aligned = featurizer.align(["LotArea", "MissingCol"], engineered, recipe)
    assert aligned["MissingCol"] == 0.0 or aligned["MissingCol"] == recipe["medians"].get("MissingCol", 0.0)


def test_predict_success_and_failure(client, api_key, db):
    # success path: mock recipe + model
    fake_recipe = {
        "passthrough_numeric": ["x"],
        "onehot": {},
        "transforms": [],
        "aggs": [],
        "agg_values": {},
        "medians": {},
        "feature_columns": ["x"],
    }

    with patch("app.serving.featurizer.load_recipe", return_value=fake_recipe), patch(
        "app.serving.featurizer.featurize", return_value={"x": 2.0}
    ), patch("app.serving.featurizer.align", return_value={"x": 2.0}), patch(
        "app.serving.service.get_model"
    ) as mock_get, patch("app.serving.service.resolve_target") as mock_resolve:
        mock_model = type("M", (), {"predict": lambda self, X: [99.0], "feature_names_in_": ["x"]})()
        mock_get.return_value = (mock_model, ("house-price-sk", 1))
        mock_resolve.return_value = {"name": "house-price-sk", "version": 1, "run_id": "r1"}
        r = client.post(
            "/predict", headers={"Authorization": f"Bearer {api_key}"}, json={"features": {"x": 2}}
        )
        assert r.status_code == 200
        assert r.json()["prediction"] == pytest.approx(99.0)
        from app.core.models import Prediction

        assert db.query(Prediction).count() >= 1

    # failure path: LookupError -> 404 when no model registered
    with patch("app.serving.featurizer.load_recipe", return_value=fake_recipe), patch(
        "app.serving.featurizer.featurize", return_value={"x": 1.0}
    ), patch("app.serving.service.resolve_target", side_effect=LookupError("no model")), patch(
        "app.serving.service.get_model", side_effect=LookupError("no model")
    ):
        r = client.post(
            "/predict", headers={"Authorization": f"Bearer {api_key}"}, json={"features": {"x": 1}}
        )
        assert r.status_code == 404


def test_nn_scaler_cached():
    import pickle
    import tempfile

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from app.serving import service as svc

    svc._scaler_cache.clear()
    # Use a real picklable scaler
    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with patch("mlflow.tracking.MlflowClient") as MockClient:
        inst = MockClient.return_value
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pickle.dump(scaler, tf)
            tf_path = tf.name
        inst.download_artifacts.return_value = tf_path

        s1 = svc._nn_scaler("house-price-nn", 1, "r1")
        s2 = svc._nn_scaler("house-price-nn", 1, "r1")
        assert s1 is s2
        assert inst.download_artifacts.call_count == 1
