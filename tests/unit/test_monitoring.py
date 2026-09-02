"""Drift threshold and window boundaries, retrain success/failure, history."""
from unittest.mock import patch


def _make_dataset_with_stats(db, mean=1000, std=200):
    from app.core.models import Dataset, DatasetVersion

    ds = Dataset(name=f"drift-test-{mean}-{std}-{id(db)}")
    db.add(ds)
    db.commit()
    dv = DatasetVersion(
        dataset_id=ds.id,
        version=1,
        dvc_md5="m" * 32,
        storage_key="k",
        original_filename="train.csv",
        column_stats={"GrLivArea": {"mean": mean, "std": std, "dtype": "float64"}, "LotArea": {"mean": 5000, "std": 1000, "dtype": "float64"}},
        n_rows=100,
        n_cols=5,
        target_column="SalePrice",
    )
    db.add(dv)
    db.commit()
    return ds, dv


def _add_predictions(db, dv, n, grlivarea_value=1000, lotarea_value=5000):
    from app.core.models import Prediction

    for _ in range(n):
        db.add(
            Prediction(
                model_name="house-price-sk",
                model_version=1,
                source="live",
                features={"GrLivArea": grlivarea_value, "LotArea": lotarea_value},
                prediction=100_000,
            )
        )
    db.commit()


def test_drift_no_trigger_when_under_window(client, api_key, db):
    ds, dv = _make_dataset_with_stats(db, mean=1000, std=200)
    _add_predictions(db, dv, n=5, grlivarea_value=2000)  # drifted but < min_window=20
    from app.monitoring import service as mon

    chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
    assert chk.verdict == "ok"
    assert not chk.triggered_retrain


def test_drift_triggers_at_z_gt_2(client, api_key, db):
    ds, dv = _make_dataset_with_stats(db, mean=1000, std=100)
    # live mean 1300 => z = |1000-1300|/100 = 3.0 > 2.0 -> drift
    _add_predictions(db, dv, n=25, grlivarea_value=1300)
    from unittest.mock import patch
    from app.core.models import TrainingRun

    def _fake_submit(db_, dataset_id=None, dataset_version=None, n_trials=15):
        tr = TrainingRun(argo_name=f"fake-{id(db_)}", status="submitted", dataset_version_id=dv.id, params={"n_trials": n_trials})
        db_.add(tr)
        db_.commit()
        db_.refresh(tr)
        return tr

    with patch("app.monitoring.service.training_service.submit_run", side_effect=_fake_submit) as mock_submit:
        from app.monitoring import service as mon

        chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
        assert chk.verdict == "drift"
        assert chk.triggered_retrain
        assert chk.training_run_id is not None
        assert "GrLivArea" in chk.features_drifted["features"]


def test_drift_not_triggered_when_within_threshold(db):
    ds, dv = _make_dataset_with_stats(db, mean=1000, std=200)
    # live mean 1010 => z=0.05 <2.0
    _add_predictions(db, dv, n=25, grlivarea_value=1010)
    with patch("app.monitoring.service.training_service.submit_run") as mock_submit:
        from app.monitoring import service as mon

        chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
        assert chk.verdict == "ok"
        assert not chk.triggered_retrain
        mock_submit.assert_not_called()


def test_drift_continue_on_missing_feature_not_break(db):
    """Missing feature on one row should not abort column (break->continue fix)."""
    ds, dv = _make_dataset_with_stats(db, mean=1000, std=100)
    from app.core.models import Prediction, TrainingRun

    # One row missing GrLivArea, others drifted
    for i in range(25):
        feats = {"LotArea": 5000} if i == 0 else {"GrLivArea": 1300, "LotArea": 5000}
        db.add(Prediction(model_name="house-price-sk", model_version=1, features=feats, prediction=1, source="live"))
    db.commit()

    def _fake_submit(db_, dataset_id=None, dataset_version=None, n_trials=15):
        tr = TrainingRun(argo_name=f"fake2-{id(db_)}", status="submitted", dataset_version_id=dv.id, params={"n_trials": n_trials})
        db_.add(tr)
        db_.commit()
        db_.refresh(tr)
        return tr

    with patch("app.monitoring.service.training_service.submit_run", side_effect=_fake_submit) as mock_submit:
        from app.monitoring import service as mon

        chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
        # Should still detect drift (24/25 values present >90% rule) despite one missing
        assert chk.verdict == "drift"


def test_retrain_failure_persisted(db):
    ds, dv = _make_dataset_with_stats(db, mean=1000, std=100)
    _add_predictions(db, dv, n=25, grlivarea_value=1500)
    with patch("app.monitoring.service.training_service.submit_run", side_effect=RuntimeError("argo down")):
        from app.monitoring import service as mon

        chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
        assert chk.verdict == "drift"
        assert not chk.triggered_retrain
        assert chk.training_run_id is None
        assert "retrain_error" in chk.features_drifted or "retrain_error" in chk.features_drifted.get("details", {})


def test_dataset_id_threading_not_hardcoded(db):
    # Create dataset id !=1 and verify submit_run gets correct dataset_id
    ds, dv = _make_dataset_with_stats(db)
    _add_predictions(db, dv, n=25, grlivarea_value=5000)  # z large
    from app.core.models import TrainingRun

    def _fake_submit(db_, dataset_id=None, dataset_version=None, n_trials=15):
        tr = TrainingRun(argo_name=f"fake3-{id(db_)}", status="submitted", dataset_version_id=dv.id, params={"n_trials": n_trials, "dataset_id": dataset_id})
        db_.add(tr)
        db_.commit()
        db_.refresh(tr)
        return tr

    with patch("app.monitoring.service.training_service.submit_run", side_effect=_fake_submit) as mock_submit:
        from app.monitoring import service as mon

        chk = mon.check_drift(db, dataset_version_id=dv.id, min_window=20)
        assert mock_submit.called
        _, kwargs = mock_submit.call_args
        # Should pass dataset_id matching dv.dataset_id, not hard-coded 1
        assert kwargs.get("dataset_id", mock_submit.call_args.args[1] if len(mock_submit.call_args.args) > 1 else None) == ds.id or mock_submit.call_args.args[1] == ds.id


def test_drift_history_endpoint(client, db, api_key):
    ds, dv = _make_dataset_with_stats(db)
    from app.monitoring import service as mon

    with patch("app.monitoring.service.training_service.submit_run"):
        mon.check_drift(db, dataset_version_id=dv.id, min_window=5)
    r = client.get("/monitoring/drift-checks")
    assert r.status_code == 200
    assert len(r.json()) >= 1
