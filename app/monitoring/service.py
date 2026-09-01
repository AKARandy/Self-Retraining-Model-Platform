"""Drift detection: compare the trailing live-prediction window against the
reference column stats captured at dataset-registration time. Stats-based
z-drift on numeric features (the plan's Evidently/stats option, stats branch)."""
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.models import DriftCheck, Prediction
from ..data import dvc_io
from ..training import service as training_service

Z_THRESHOLD = 0.5  # |ref_mean - window_mean| / ref_std
WINDOW_MINUTES = 60


def _reference_stats(db: Session, dataset_version_id: int) -> dict:
    from ..core.models import DatasetVersion

    row = db.query(DatasetVersion).filter_by(id=dataset_version_id).first()
    if row is None or not row.column_stats:
        return {}
    return {
        col: s
        for col, s in row.column_stats.items()
        if isinstance(s, dict) and s.get("mean") is not None and (s.get("std") or 0) > 0
    }


def check_drift(db: Session, dataset_version_id: int = 1, min_window: int = 20) -> DriftCheck:
    ref = _reference_stats(db, dataset_version_id)
    since = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)
    rows = db.query(Prediction).filter(Prediction.created_at >= since).all()

    verdict = {"drifted": [], "details": {}, "n_predictions": len(rows)}

    if len(rows) >= min_window and ref:
        for col, stats in ref.items():
            vals = []
            for r in rows:
                v = (r.features or {}).get(col)
                if v is None:
                    break
                vals.append(float(v))
            if len(vals) < len(rows) * 0.9:  # feature missing from traffic
                continue
            live_mean = float(np.mean(vals))
            z = abs(stats["mean"] - live_mean) / stats["std"]
            verdict["details"][col] = {
                "ref_mean": round(stats["mean"], 2),
                "live_mean": round(live_mean, 2),
                "z": round(z, 3),
            }
            if z > Z_THRESHOLD:
                verdict["drifted"].append(col)

    has_drift = bool(verdict["drifted"])
    check = DriftCheck(
        verdict="drift" if has_drift else "ok",
        features_drifted={
            "features": verdict["drifted"],
            "details": {k: v for k, v in verdict["details"].items() if k in verdict["drifted"]},
            "n_predictions": verdict["n_predictions"],
        },
        triggered_retrain=False,
    )

    if has_drift:
        try:
            run = training_service.submit_run(db, dataset_id=1, dataset_version=None, n_trials=15)
            check.training_run_id = run.id
            check.triggered_retrain = True
        except Exception as e:
            verdict["details"]["retrain_error"] = str(e)[:300]

    db.add(check)
    db.commit()
    db.refresh(check)
    return check


def latest_check(db: Session) -> DriftCheck | None:
    return db.query(DriftCheck).order_by(DriftCheck.id.desc()).first()
