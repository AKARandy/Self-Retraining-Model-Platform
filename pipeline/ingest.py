"""Step 1: ingest — pull dataset version bytes and park them as parquet in the artifact bucket."""
import io
import sys

import pandas as pd

from common import ARTIFACT_BUCKET, TARGET, WORKFLOW_NAME, fetch_dataset_content, md5_bytes, put_artifact, put_json


def main() -> None:
    dataset_id = int(sys.argv[sys.argv.index("--dataset_id") + 1]) if "--dataset_id" in sys.argv else 1
    dataset_version = int(sys.argv[sys.argv.index("--dataset_version") + 1]) if "--dataset_version" in sys.argv else 1

    content = fetch_dataset_content(dataset_id, dataset_version)
    df = pd.read_csv(io.BytesIO(content))
    if TARGET not in df.columns:
        raise SystemExit(f"fatal: target column {TARGET!r} not in dataset (cols: {list(df.columns)[:10]}...)")

    key = f"houses/{WORKFLOW_NAME}/raw.parquet"
    put_artifact(key, df.to_parquet(index=False))
    put_json(
        f"houses/{WORKFLOW_NAME}/ingest.json",
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "content_md5": md5_bytes(content),
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "target": TARGET,
            "artifact": f"s3://{ARTIFACT_BUCKET}/{key}",
        },
    )
    print(f"ingested {df.shape[0]} rows x {df.shape[1]} cols -> {key}")


if __name__ == "__main__":
    main()
