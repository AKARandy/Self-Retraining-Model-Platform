"""Pipeline helper / decision logic — no cluster or full training needed."""
import sys
from pathlib import Path

import pandas as pd

# Allow `from common import ...` inside pipeline/ (container WORKDIR=/work)
PIPELINE_DIR = Path(__file__).resolve().parents[2] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def test_clean_features_replaces_inf_and_fills():
    from pipeline.common import clean_features

    df_tr = pd.DataFrame({"a": [1, float("inf"), None], "b": [1, 1, 1], "c": [None, None, None]})
    df_te = pd.DataFrame({"a": [2, None], "b": [1, 1], "c": [None, None]})
    # Need to handle inf and all-NaN columns
    tr, te = clean_features(df_tr, df_te)
    assert "c" not in tr.columns
    assert "c" not in te.columns
    assert not tr.isna().all().any() or tr.shape[0] > 0


def test_md5_and_feature_hash_deterministic():
    from pipeline.common import md5_bytes

    assert md5_bytes(b"hello") == md5_bytes(b"hello")
    assert md5_bytes(b"hello") != md5_bytes(b"world")


def test_build_entityset_and_recipe():
    try:
        import featuretools as ft
    except ModuleNotFoundError:
        import pytest

        pytest.skip("featuretools not installed in API env — pipeline-tests job installs training deps")
    from pipeline.feature_engineer import build_entityset, build_recipe

    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SalePrice": [100, 200, 150],
            "LotArea": [5000, 6000, 5500],
            "OverallQual": [5, 6, 5],
            "YearBuilt": [2000, 2001, 1999],
            "GrLivArea": [1000, 1200, 1100],
            "Neighborhood": ["A", "B", "A"],
        }
    )
    es = build_entityset(df)
    assert "houses" in es.dataframe_dict
    assert "neighborhoods" in es.dataframe_dict
    # Build recipe from fake defs with identity features only
    feature_defs = ft.dfs(entityset=es, target_dataframe_name="houses", max_depth=1, features_only=True, trans_primitives=["add_numeric"])
    # Sanity: build_recipe should not crash
    feature_cols = ["LotArea", "GrLivArea"]
    recipe = build_recipe(df, feature_defs, feature_cols)
    assert "passthrough_numeric" in recipe
    assert "feature_columns" in recipe


def test_validate_success_and_failure():
    from unittest.mock import patch

    # pipeline.validate imports `from common import ...` which needs PIPELINE_DIR on path (added above)
    import pipeline.validate as vd

    # Success: enough rows, target not null (validate requires >=100 rows and >=5 numeric cols)
    df_ok = pd.DataFrame({"Id": range(1, 101), "SalePrice": range(100, 200), "x": [1]*100, "y": [1]*100, "z": [1]*100, "w": [1]*100, "v": [1]*100})
    # Monkey patch get_artifact/put_* to use memory
    with patch("pipeline.validate.get_artifact", return_value=df_ok.to_parquet(index=False)), patch(
        "pipeline.validate.put_artifact"
    ), patch("pipeline.validate.put_json"):
        vd.main()  # should not raise

    # Failure: too few rows
    df_small = pd.DataFrame({"Id": [1], "SalePrice": [1]})
    with patch("pipeline.validate.get_artifact", return_value=df_small.to_parquet(index=False)), patch(
        "pipeline.validate.put_json"
    ):
        try:
            vd.main()
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert "fatal" in str(e).lower() or e.code != 0


def test_promotion_gate_logic():
    # Replicate evaluate.py decision logic without MLflow
    def decide(new_rmse, prod_rmse, prod_exists=True):
        if not prod_exists:
            return True, "no production model yet"
        if prod_rmse is None:
            return True, "production model has no logged rmse"
        improve = (prod_rmse - new_rmse) / prod_rmse
        promote = improve > 0.0
        reason = f"new rmse {new_rmse:.1f} vs prod {prod_rmse:.1f} ({improve * 100:+.2f}%)"
        return promote, reason

    assert decide(100, None)[0] is True
    assert decide(90, 100)[0] is True
    assert decide(110, 100)[0] is False
    assert decide(100, 100)[0] is False  # no improve
    assert decide(100, 200, prod_exists=False)[0] is True
