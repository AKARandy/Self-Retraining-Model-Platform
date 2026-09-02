"""Step 2: validate — sanity checks on the raw data; exit 1 on fatal problems."""
import io

import pandas as pd
from common import TARGET, WORKFLOW_NAME, get_artifact, put_artifact, put_json


def main() -> None:
    key = f"houses/{WORKFLOW_NAME}/raw.parquet"
    df = pd.read_parquet(io.BytesIO(get_artifact(key)))

    report = {"checks": [], "dropped_columns": [], "fatal": []}

    ok_rows = len(df) >= 100
    report["checks"].append({"check": "min_rows", "ok": ok_rows, "n_rows": len(df)})
    if not ok_rows:
        report["fatal"].append("dataset too small")

    target_ok = not df[TARGET].isna().any()
    report["checks"].append({"check": "target_not_null", "ok": target_ok})
    if not target_ok:
        report["fatal"].append("target has missing values")

    # entirely-null predictors carry no signal; drop and report them
    for col in list(df.columns):
        if col != TARGET and df[col].isna().all():
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)

    n_numeric = int(df.select_dtypes("number").shape[1])
    report["checks"].append({"check": "numeric_columns", "ok": n_numeric >= 5, "n_numeric": n_numeric})
    report["n_rows"] = int(df.shape[0])
    report["n_columns_after"] = int(df.shape[1])

    if report["fatal"]:
        put_json(f"houses/{WORKFLOW_NAME}/validation.json", report)
        raise SystemExit(f"fatal: {'; '.join(report['fatal'])}")

    put_artifact(key, df.to_parquet(index=False))  # overwrite with the cleaned frame
    put_json(f"houses/{WORKFLOW_NAME}/validation.json", report)
    print("validation ok:", report["n_rows"], "rows,", report["n_columns_after"], "cols")


if __name__ == "__main__":
    main()
