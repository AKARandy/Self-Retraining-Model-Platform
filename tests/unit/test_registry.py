"""Registry listing/current/promotion/card via mocked MlflowClient."""
from unittest.mock import MagicMock, patch


def _mock_version(name="house-price-sk", version="1", stage="None", run_id="r1"):
    mv = MagicMock()
    mv.name = name
    mv.version = version
    mv.current_stage = stage
    mv.run_id = run_id
    mv.creation_timestamp = 1_700_000_000_000
    mv.source = "s3://mlflow-artifacts/x"
    mv.status = "READY"
    return mv


def test_list_models_and_versions(client):
    mv = _mock_version()
    rm = MagicMock()
    rm.name = "house-price-sk"
    rm.latest_versions = [mv]

    with patch("app.registry.service.MlflowClient") as MockClient:
        inst = MockClient.return_value
        inst.search_registered_models.return_value = [rm]
        inst.search_model_versions.return_value = [mv]
        r = client.get("/registry/models")
        assert r.status_code == 200
        assert r.json()[0]["name"] == "house-price-sk"
        r2 = client.get("/registry/models/house-price-sk/versions")
        assert r2.status_code == 200
        assert r2.json()[0]["version"] == 1


def test_production_version_lookup(client):
    mv_prod = _mock_version(stage="Production", version="2", run_id="r-prod")
    mv_staging = _mock_version(stage="None", version="1", run_id="r1")
    with patch("app.registry.service.MlflowClient") as MockClient:
        inst = MockClient.return_value
        inst.search_model_versions.return_value = [mv_prod, mv_staging]
        r = client.get("/registry/models/house-price-sk/current")
        assert r.status_code == 200
        assert r.json()["version"] == 2
        assert r.json()["run_id"] == "r-prod"


def test_promote_success_audited(client, api_key, db):
    mv = _mock_version(version="3", stage="None", run_id="r3")
    with patch("app.registry.service.MlflowClient") as MockClient, patch(
        "app.registry.routes.record"
    ) as mock_record:
        inst = MockClient.return_value
        inst.get_model_version.return_value = mv
        inst.transition_model_version_stage.return_value = None
        r = client.post(
            "/registry/promote",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"name": "house-price-sk", "version": 3},
        )
        assert r.status_code == 200


def test_promote_missing_returns_404(client, api_key):
    with patch("app.registry.service.MlflowClient") as MockClient:
        from mlflow.exceptions import MlflowException

        inst = MockClient.return_value
        inst.get_model_version.side_effect = MlflowException("not found")
        r = client.post(
            "/registry/promote",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"name": "house-price-sk", "version": 999},
        )
        assert r.status_code == 404


def test_model_card_renders(client):
    mv = _mock_version(version="1", stage="Production", run_id="r1")
    run = MagicMock()
    run.info.run_id = "r1"
    run.info.experiment_id = "1"
    run.data.metrics = {"rmse": 1234.5, "r2": 0.85}
    run.data.params = {"family": "histgb"}

    with patch("app.registry.service.MlflowClient") as MockClient, patch(
        "app.registry.card.dvc_io.s3_client"
    ) as mock_s3, patch("app.registry.card.dvc_io.get_artifact", return_value='{"top_features": []}'):
        inst = MockClient.return_value
        inst.get_model_version.return_value = mv
        inst.get_run.return_value = run
        mock_s3.return_value.list_objects_v2.return_value = {"Contents": []}
        r = client.get("/registry/models/house-price-sk/versions/1/card")
        assert r.status_code == 200
        assert "Model card" in r.text
