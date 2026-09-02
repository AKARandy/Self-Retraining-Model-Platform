"""Single API key gate on state-changing routes. Rejected attempts are audited."""
from fastapi import HTTPException, Request
from sqlalchemy.orm import sessionmaker

from .audit import record
from .config import settings
from .db import SessionLocal


def require_api_key(request: Request) -> None:
    expected = f"Bearer {settings.api_key}"
    supplied = request.headers.get("Authorization", "")
    db: sessionmaker = SessionLocal()
    try:
        if supplied != expected:
            try:
                record(
                    db,
                    action=f"{request.method} {request.url.path}",
                    resource=request.url.path,
                    allowed=False,
                    detail={"reason": "missing or invalid API key"},
                )
            except Exception:
                # Audit failure must not hide 401; log and continue to raise
                import logging

                logging.getLogger(__name__).exception("audit failed for rejected request")
            raise HTTPException(401, "missing or invalid API key")
    finally:
        db.close()
