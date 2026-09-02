"""Argo submission payloads, status sync, missing dataset, remote failures."""
from unittest.mock import MagicMock, patch


def test_submit_missing_dataset_returns_404(client, api_key):
    with patch("app.training.service.httpx.post") as mock_post:
        # submit_run should raise ValueError before httpx — so mock not called
        r = client.post(
            "/train-runs",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"dataset_id": 9999, "n_trials": 5},
        )
        assert r.status_code == 404


def test_submit_payload_contains_params(client, api_key, db):
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name="train-payload-test")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(
        dataset_id=ds.id,
        version=1,
        dvc_md5="x" * 32,
        storage_key="k",
        original_filename="train.csv",
        column_stats={},
    )
    db.add(dv)
    db.commit()

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "metadata": {"name": "train-pipeline-abc", "uid": "uid-123"},
        "status": {"phase": "Running"},
    }
    fake_resp.raise_for_status = MagicMock()

    with patch("app.training.service.httpx.post", return_value=fake_resp) as mock_post:
        r = client.post(
            "/train-runs",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"dataset_id": ds.id, "dataset_version": 1, "n_trials": 5},
        )
        assert r.status_code == 200, r.text
        args, kwargs = mock_post.call_args
        body = kwargs["json"]
        # submitOptions.parameters is List[str] "name=value"
        params_list = body["submitOptions"]["parameters"]
        params = dict(p.split("=", 1) for p in params_list)
        assert params["dataset_id"] == str(ds.id)
        assert params["n_trials"] == "5"


def test_submit_remote_failure_returns_502(client, api_key, db):
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name="train-502-test")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(dataset_id=ds.id, version=1, dvc_md5="y" * 32, storage_key="k2", original_filename="train.csv", column_stats={})
    db.add(dv)
    db.commit()

    with patch("app.training.service.httpx.post", side_effect=Exception("argo down")):
        r = client.post(
            "/train-runs",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"dataset_id": ds.id, "n_trials": 5},
        )
        assert r.status_code == 502


def test_sync_status_maps_phase(client, api_key, db):
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name="sync-test")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(dataset_id=ds.id, version=1, dvc_md5="z" * 32, storage_key="k3", original_filename="train.csv", column_stats={})
    db.add(dv)
    db.commit()
    # Submit fake run row via mocked Argo post
    fake_post = MagicMock()
    fake_post.status_code = 200
    fake_post.json.return_value = {"metadata": {"name": "wf-1", "uid": "uid-1"}, "status": {"phase": "Running"}}
    fake_post.raise_for_status = MagicMock()
    with patch("app.training.service.httpx.post", return_value=fake_post):
        r = client.post(
            "/train-runs",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"dataset_id": ds.id, "n_trials": 5},
        )
        run_id = r.json()["id"]

    # Mock GET for sync — Succeeded
    fake_get = MagicMock()
    fake_get.status_code = 200
    fake_get.json.return_value = {"status": {"phase": "Succeeded", "nodes": {}}}
    with patch("app.training.service.httpx.get", return_value=fake_get):
        r2 = client.get(f"/train-runs/{run_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "Succeeded"
