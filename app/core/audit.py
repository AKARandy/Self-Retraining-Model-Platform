from sqlalchemy.orm import Session

from .models import AuditLog


def record(db: Session, action: str, resource: str, allowed: bool, detail: dict | None = None) -> None:
    """One row per write attempt (allowed or rejected)."""
    db.add(AuditLog(action=action, resource=resource, allowed=allowed, detail=detail))
    db.commit()
