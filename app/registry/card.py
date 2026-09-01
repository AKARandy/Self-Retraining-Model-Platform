"""Model card rendering: run metrics + params + SHAP summary + lineage."""
import io
import json
from datetime import datetime, timezone

from ..data import dvc_io

CARD_TEMPLATE = """# Model card — {name} v{version}

- **Stage:** {stage}
- **MLflow run:** [{run_id}]({tracking_uri}/#/experiments/{exp_id}/runs/{run_id})
- **Registered:** {registered_at}
- **Flavor:** sklearn (tree ensemble) / Optuna-tuned

## Test metrics
| metric | value |
|---|---|
{metrics_rows}

## Hyperparameters (winner)
```text
{params}
```

## Explainability — SHAP top features
{shap_section}

## Data lineage
- Feature pipeline: Featuretools DFS (`add_numeric`, `multiply_numeric` transforms; neighborhood-level `mean`/`std`/`sum`/`count` aggregations), max_depth=1
- Training orchestration: Argo WorkflowTemplate `train-pipeline` (ingest → validate → feature-engineer → train(Optuna) → evaluate → shap → promote)
- Model promotion: conditional on test RMSE vs current Production (evaluated in-pipeline)

_Generated {generated_at} — metrics and SHAP values are read live from the run's logged artifacts._
"""


def _latest_shap() -> dict | None:
    s3 = dvc_io.s3_client()
    resp = s3.list_objects_v2(Bucket=dvc_io.settings.bucket_artifacts, Prefix="houses/")
    keys = [
        o
        for o in resp.get("Contents", [])
        if o["Key"].endswith("shap_top20.json")
    ]
    if not keys:
        return None
    latest = max(keys, key=lambda o: o["LastModified"])
    return json.loads(dvc_io.get_artifact(latest["Key"]))


def render_card(mv, run, tracking_uri: str) -> str:
    metrics = run.data.metrics
    metrics_rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in sorted(metrics.items())) or "| (none logged) | |"

    shap = _latest_shap()
    if shap and shap.get("top_features"):
        tops = "\n".join(
            f"{i + 1}. `{t['feature']}` — mean |SHAP| {t['mean_abs_shap']:.1f}"
            for i, t in enumerate(shap["top_features"][:10])
        )
        shap_section = tops
    else:
        shap_section = "_SHAP artifacts not yet generated for this model._"

    params = "\n".join(f"{k}={v}" for k, v in sorted(run.data.params.items())) or "(none)"
    exp_id = run.info.experiment_id
    registered_at = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc).isoformat()

    return CARD_TEMPLATE.format(
        name=mv.name,
        version=mv.version,
        stage=mv.current_stage or "None",
        run_id=run.info.run_id,
        exp_id=exp_id,
        tracking_uri=tracking_uri,
        registered_at=registered_at,
        metrics_rows=metrics_rows,
        params=params,
        shap_section=shap_section,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
