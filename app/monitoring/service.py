"""Drift detection: compare the trailing live-prediction window against the
reference column stats captured at dataset-registration time. Stats-based
z-drift on numeric features (the plan's Evidently/stats option, stats branch)."""
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..core.models import DriftCheck, Prediction
from ..training import service as training_service

Z_THRESHOLD = 2.0  # |ref_mean - window_mean| / ref_std; 2.0 ≈ 5% two-tailed; 0.5 fired on ~62% noise (see VERTWOPLAN §19.3).
# Honest limitation: family-wise false-positive compounds across independent z-tests —
# e.g. ~26% chance of ≥1 spurious drift across ~6 numeric features at 5% each.
# Mitigation deferred (Bonferroni / require ≥2 breaches); see VERTWOPLAN §19.3.
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
    # Only live predictions (with features) — batch predictions have features=NULL and would blind drift
    rows = db.query(Prediction).filter(Prediction.created_at >= since, Prediction.source == "live").all()

    verdict = {"drifted": [], "details": {}, "n_predictions": len(rows)}

    if len(rows) >= min_window and ref:
        for col, stats in ref.items():
            vals = []
            for r in rows:
                v = (r.features or {}).get(col)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
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
        # Debounce: if a run for this dataset is still submitted/running, don't spam
        from ..core.models import TrainingRun as _TR

        _recent = (
            db.query(_TR)
            .filter(_TR.dataset_version_id.isnot(None))
            .order_by(_TR.id.desc())
            .first()
        )
        # Simple debounce: if latest run is still not terminal, reuse it
        _debounced = False
        if _recent and _recent.status not in ("Succeeded", "Failed", "Error") and _recent.finished_at is None:
            # Treat as triggered but point to existing run
            check.training_run_id = _recent.id
            check.triggered_retrain = False
            _debounced = True

        if not _debounced:
            try:
                from ..core.models import DatasetVersion as _DV

                _dv = db.query(_DV).filter_by(id=dataset_version_id).first()
                _ds_id = _dv.dataset_id if _dv else 1
                run = training_service.submit_run(db, dataset_id=_ds_id, dataset_version=None, n_trials=15)
                check.training_run_id = run.id
                check.triggered_retrain = True
            except Exception as e:
                # Persist the error so the stored drift check records why retrain failed.
                err = str(e)[:300]
                verdict["details"]["retrain_error"] = err
                # Also include in persisted features_drifted (copy current drift details).
                check.features_drifted = {
                    **check.features_drifted,
                    "retrain_error": err,
                }
                flag_modified(check, "features_drifted")

    # If retrain failed, ensure features_drifted already contains retrain_error before add
    if "retrain_error" in verdict["details"] and "retrain_error" not in (check.features_drifted or {}):
        check.features_drifted = {**(check.features_drifted or {}), "retrain_error": verdict["details"]["retrain_error"]}
        flag_modified(check, "features_drifted")

    db.add(check)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(check)
    return check


def latest_check(db: Session) -> DriftCheck | None:
    return db.query(DriftCheck).order_by(DriftCheck.id.desc()).first()
