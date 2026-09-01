"""Step 8: promote — act on evaluate's decision by moving the candidate to Production."""
import os

import mlflow

from common import WORKFLOW_NAME, get_json

from mlflow.tracking import MlflowClient


def main() -> None:
    decision = get_json(f"houses/{WORKFLOW_NAME}/decision.json")
    cand = decision["candidate"]

    if not decision["promote"]:
        print("no promotion:", decision["reason"])
        return

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    c = MlflowClient()
    c.transition_model_version_stage(
        name=cand["name"],
        version=str(cand["version"]),
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"promoted {cand['name']} v{cand['version']} to Production ({decision['reason']})")


if __name__ == "__main__":
    main()
