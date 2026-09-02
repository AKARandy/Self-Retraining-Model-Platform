"""Integration: full app via TestClient on disposable Postgres (when available) or SQLite fallback."""
from unittest.mock import MagicMock, patch


def test_health_and_list_datasets(client):
    assert client.get("/health").status_code == 200
    assert client.get("/datasets").status_code == 200


def test_train_run_lifecycle_mocked(client, api_key, db):
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name="int-train-test")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(dataset_id=ds.id, version=1, dvc_md5="a" * 32, storage_key="k", original_filename="train.csv", column_stats={})
    db.add(dv)
    db.commit()

    fake_post = MagicMock()
    fake_post.status_code = 200
    fake_post.json.return_value = {"metadata": {"name": "wf-int-1", "uid": "uid-int"}, "status": {"phase": "Running"}}
    fake_post.raise_for_status = MagicMock()
    fake_get = MagicMock()
    fake_get.status_code = 200
    fake_get.json.return_value = {"status": {"phase": "Succeeded", "nodes": {}}}

    with patch("app.training.service.httpx.post", return_value=fake_post), patch(
        "app.training.service.httpx.get", return_value=fake_get
    ):
        r = client.post("/train-runs", headers={"Authorization": f"Bearer {api_key}"}, json={"dataset_id": ds.id, "n_trials": 2})
        assert r.status_code == 200
        run_id = r.json()["id"]
        r2 = client.get(f"/train-runs/{run_id}")
        assert r2.status_code == 200


def test_audit_log_covers_rejected_and_allowed(client, api_key, db):
    # rejected
    client.post("/train-runs", json={"dataset_id": 1})
    from app.core.models import AuditLog

    rejected = db.query(AuditLog).filter_by(allowed=False).count()
    assert rejected >= 1
    # allowed — promote mocked
    mv = MagicMock()
    mv.name = "house-price-sk"
    mv.version = "1"
    mv.run_id = "r1"
    with patch("app.registry.service.MlflowClient") as MockClient:
        inst = MockClient.return_value
        inst.get_model_version.return_value = mv
        inst.transition_model_version_stage.return_value = None
        r = client.post("/registry/promote", headers={"Authorization": f"Bearer {api_key}"}, json={"name": "house-price-sk", "version": 1})
        assert r.status_code == 200
    allowed = db.query(AuditLog).filter_by(allowed=True).count()
    assert allowed >= 1
