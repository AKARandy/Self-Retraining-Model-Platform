"""Rebuild the exact engineered feature vector from one raw feature dict,
using the recipe emitted by the feature-engineer step."""
import json
import time

from ..data.dvc_io import get_artifact

_RECIPE = None
_LOADED_AT = 0.0
_TTL_SECONDS = 60


def load_recipe() -> dict:
    global _RECIPE, _LOADED_AT
    if _RECIPE is None or (time.time() - _LOADED_AT) > _TTL_SECONDS:
        pointer = json.loads(get_artifact("houses/recipe/latest.json"))
        _RECIPE = json.loads(get_artifact(pointer["recipe_key"]))
        _LOADED_AT = time.time()
    return _RECIPE


def featurize(raw: dict, recipe: dict) -> dict:
    out: dict = {}

    for col in recipe["passthrough_numeric"]:
        if col in raw and raw[col] is not None:
            try:
                out[col] = float(raw[col])
            except (TypeError, ValueError):
                pass

    # one-hot categoricals: one column per known training level,
    # plus the _nan dummy that pd.get_dummies(dummy_na=True) created in training
    for col, levels in recipe["onehot"].items():
        val = raw.get(col)
        for lv in levels:
            out[f"{col}_{lv}"] = 1.0 if str(val) == lv else 0.0
        missing = val is None or (isinstance(val, float) and val != val)
        out[f"{col}_nan"] = 1.0 if missing else 0.0

    for t in recipe["transforms"]:
        a, b = raw.get(t["a"]), raw.get(t["b"])
        if a is None or b is None:
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        out[t["column"]] = a + b if t["op"] == "add" else a * b

    nb = str(raw.get("Neighborhood", ""))
    nb_vals = recipe["agg_values"].get(nb, {})
    for a in recipe["aggs"]:
        v = nb_vals.get(a["column"])
        if v is not None:
            out[a["column"]] = v

    # NaN handling identical to training: fill with training-set medians
    for col, med in recipe.get("medians", {}).items():
        out.setdefault(col, med)
    return out


def align(frame_columns: list[str], engineered: dict, recipe: dict) -> dict:
    """Model-ready vector: model column order, missing -> 0."""
    return {c: engineered.get(c, recipe.get("medians", {}).get(c, 0.0)) for c in frame_columns}
