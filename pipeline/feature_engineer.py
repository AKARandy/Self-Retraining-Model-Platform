"""Step 3: feature engineering — Featuretools DFS over a houses/neighborhood entity pair,
feature-set provenance row written straight into Postgres, plus a serving recipe that
lets the API rebuild the exact engineered feature vector from one raw row."""
import hashlib
import io
import json

import featuretools as ft
import pandas as pd

from common import TARGET, WORKFLOW_NAME, db_engine, get_artifact, put_artifact, put_json

ID_COL = "Id"

TRANSFORM_PRIMITIVES = ["add_numeric", "multiply_numeric"]
AGG_PRIMITIVES = ["mean", "std", "sum", "count"]

# numeric columns used to build the neighborhood child entity
AGG_COLUMNS = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea"]


def build_entityset(df: pd.DataFrame) -> ft.EntitySet:
    df = df.copy()

    # parent entity: one row per house
    houses = df.drop_duplicates(subset=[ID_COL]).copy()
    for c in houses.columns:
        if c not in (ID_COL, TARGET) and houses[c].dtype == object:
            houses[c] = houses[c].astype("category")

    # child entity: one row per neighborhood with base values for aggregations
    cols = ["Neighborhood"] + [c for c in AGG_COLUMNS if c in df.columns]
    nb = df[cols].copy()
    nb["nb_id"] = nb["Neighborhood"].astype(str)
    nb = nb.drop_duplicates(subset=["nb_id"]).reset_index(drop=True)
    nb = nb.rename(columns={c: f"base_{c}" for c in AGG_COLUMNS if c in df.columns})

    es = ft.EntitySet(id="houses_es")
    es.add_dataframe(dataframe_name="houses", dataframe=houses, index=ID_COL)
    es.add_dataframe(dataframe_name="neighborhoods", dataframe=nb, index="nb_id")
    es.add_relationship("neighborhoods", "nb_id", "houses", "Neighborhood")
    return es


def sanitize(name: str) -> str:
    return str(name).replace(" ", "_").replace("[", "").replace("]", "")


def build_recipe(df: pd.DataFrame, defs: list, feature_cols: list[str]) -> dict:
    """Deterministic description of every engineered column so the API can
    reproduce them from one raw row without running Featuretools."""
    recipe = {
        "id": ID_COL,
        "target": TARGET,
        "passthrough_numeric": [],
        "onehot": {},
        "transforms": [],  # {column, op: add|mul, a, b}
        "aggs": [],        # {column, base, op} — looked up per-neighborhood in agg_values
        "agg_values": {},  # neighborhood -> {column: value}
        "medians": {},
        "feature_columns": feature_cols,
    }

    numeric_cols = [c for c in df.select_dtypes("number").columns if c not in (ID_COL, TARGET)]

    for f in defs:
        name = sanitize(f.get_name())
        if name not in feature_cols:
            continue
        prim_cls = type(getattr(f, "primitive", None)).__name__
        f_cls = type(f).__name__
        if prim_cls in ("AddNumeric", "MultiplyNumeric"):
            recipe["transforms"].append(
                {"column": name, "op": "add" if prim_cls == "AddNumeric" else "mul",
                 "a": f.base_features[0].get_name(), "b": f.base_features[1].get_name()}
            )
        elif prim_cls in ("Mean", "Std", "Sum", "Count") and f_cls == "AggregationFeature":
            recipe["aggs"].append(
                {"column": name, "base": f.base_features[0].get_name(), "op": prim_cls.lower()}
            )
        elif f_cls == "IdentityFeature":
            col = f.get_name()
            if col in numeric_cols or col == ID_COL:
                recipe["passthrough_numeric"].append(col)
            else:
                recipe["onehot"][col] = sorted(str(v) for v in df[col].dropna().unique())

    nb_col = "Neighborhood"
    for a in recipe["aggs"]:
        # child-entity base columns are named base_<col>; the raw frame has <col>
        base_raw = a["base"].removeprefix("base_")
        if base_raw not in df.columns:
            continue
        agg = {"mean": "mean", "std": "std", "sum": "sum", "count": "count"}[a["op"]]
        vals = df.groupby(nb_col)[base_raw].agg(agg)
        for nb, v in vals.items():
            recipe["agg_values"].setdefault(str(nb), {})[a["column"]] = float(v)

    return recipe


def main() -> None:
    import sys

    dataset_version_id = (
        int(sys.argv[sys.argv.index("--dataset_version_id") + 1]) if "--dataset_version_id" in sys.argv else None
    )

    raw_key = f"houses/{WORKFLOW_NAME}/raw.parquet"
    df = pd.read_parquet(io.BytesIO(get_artifact(raw_key)))
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    df.insert(0, ID_COL, range(1, len(df) + 1))

    es = build_entityset(df)
    feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="houses",
        agg_primitives=AGG_PRIMITIVES,
        trans_primitives=TRANSFORM_PRIMITIVES,
        max_depth=1,
        features_only=True,
    )
    features = ft.calculate_feature_matrix(feature_defs, entityset=es)
    # drop target and anything derived from it (add/multiply transforms leak it otherwise)
    leaked = [c for c in features.columns if TARGET in str(c)]
    if leaked:
        features = features.drop(columns=leaked)

    # re-attach target
    features = features.merge(df[[ID_COL, TARGET]], on=ID_COL, how="inner")

    # one-hot remaining categoricals
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        features = pd.get_dummies(features, columns=cat_cols, dummy_na=True)
    features.columns = [sanitize(c) for c in features.columns]
    features = features.select_dtypes(include=["number", "bool"])
    bool_cols = features.select_dtypes("bool").columns
    features[bool_cols] = features[bool_cols].astype(int)

    feature_names = [c for c in features.columns if c not in (ID_COL, TARGET)]
    payload = features.to_parquet(index=False)
    key = f"houses/{WORKFLOW_NAME}/features.parquet"
    put_artifact(key, payload)

    feature_hash = hashlib.md5(
        json.dumps(
            {"base": raw_key, "names": sorted(feature_names), "prims": TRANSFORM_PRIMITIVES + AGG_PRIMITIVES}
        ).encode()
    ).hexdigest()

    # serving recipe from the DFS definitions
    # exclude transform columns that got one-hot dummies (none — transforms are numeric)
    base_defs_names = {sanitize(f.get_name()) for f in feature_defs}
    recipe = build_recipe(df, feature_defs, base_defs_names)
    # medians of the final engineered frame (for NaN handling at serve time)
    recipe["medians"] = {c: float(features[c].median()) for c in feature_names if features[c].isna().any()}
    # final aligned column list AFTER one-hot expansion (model-ready order)
    recipe["feature_columns"] = feature_names
    recipe["primitives"] = {
        "trans_primitives": TRANSFORM_PRIMITIVES,
        "agg_primitives": AGG_PRIMITIVES,
        "agg_columns": AGG_COLUMNS,
        "max_depth": 1,
    }
    put_json(f"houses/{WORKFLOW_NAME}/recipe.json", recipe)
    # stable pointer for the serving layer
    put_json("houses/recipe/latest.json", {"recipe_key": f"houses/{WORKFLOW_NAME}/recipe.json", "workflow": WORKFLOW_NAME})

    primitives_log = recipe["primitives"]

    # provenance row in Postgres
    if dataset_version_id is not None:
        from sqlalchemy import text

        with db_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO feature_sets (dataset_version_id, feature_hash, primitives, artifact_key, n_features)"
                    " VALUES (:dv, :fh, :pr, :ak, :nf)"
                ),
                {
                    "dv": dataset_version_id,
                    "fh": feature_hash,
                    "pr": json.dumps(primitives_log),
                    "ak": f"s3://pipeline-artifacts/{key}",
                    "nf": len(feature_names),
                },
            )

    put_json(
        f"houses/{WORKFLOW_NAME}/features.json",
        {
            "artifact": f"s3://pipeline-artifacts/{key}",
            "feature_hash": feature_hash,
            "n_features": len(feature_names),
            "n_rows": int(features.shape[0]),
            "primitives": primitives_log,
        },
    )
    print(f"features: {features.shape[0]} rows x {len(feature_names)} engineered cols, hash={feature_hash[:12]}")


if __name__ == "__main__":
    main()
