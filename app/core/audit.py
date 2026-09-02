from sqlalchemy.orm import Session

from .models import AuditLog


def record(db: Session, action: str, resource: str, allowed: bool, detail: dict | None = None, commit: bool = True) -> None:
    """One row per write attempt (allowed or rejected).

    Simple fix: when commit=False, only flush so caller can commit atomically
    (e.g. prediction + audit in one transaction). Otherwise commit with rollback on failure.
    """
    db.add(AuditLog(action=action, resource=resource, allowed=allowed, detail=detail))
    if commit:
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
    else:
        try:
            db.flush()
        except Exception:
            db.rollback()
            raise
