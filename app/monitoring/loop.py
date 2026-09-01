"""Host-side drift monitor loop: run check-drift on an interval.
The pod->host:8000 direction is unreliable behind Windows Firewall, so the
periodic trigger lives next to the API. CronWorkflow copy kept as artifact."""
import os
import time

from sqlalchemy.orm import Session

from ..core.db import SessionLocal
from . import service


def main() -> None:
    interval = int(os.environ.get("MONITOR_INTERVAL_SECONDS", "1800"))
    dv_id = int(os.environ.get("MONITOR_DATASET_VERSION_ID", "1"))
    min_window = int(os.environ.get("MONITOR_MIN_WINDOW", "20"))
    print(f"drift monitor: every {interval}s, dataset_version_id={dv_id}, min_window={min_window}")
    while True:
        db: Session = SessionLocal()
        try:
            check = service.check_drift(db, dv_id, min_window)
            print(
                f"drift check #{check.id}: verdict={check.verdict} "
                f"retrain={check.triggered_retrain} run={check.training_run_id}"
            )
        except Exception as e:
            print("drift check error:", e)
        finally:
            db.close()
        time.sleep(interval)


if __name__ == "__main__":
    main()
