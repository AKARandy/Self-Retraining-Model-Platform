"""Shared fixtures — disposable Postgres DB with Alembic, mocks at boundaries.

- Never uses workstation 5433 DB or real DVC/MinIO/MLflow/Argo.
- App tests use SQLite in-memory by default (fast unit) and a disposable
  Postgres when DATABASE_URL_TEST / TEST_DATABASE_URL is provided (integration).
  CI sets TEST_DATABASE_URL to the postgres:16 service on 5432.
- Migrations: alembic upgrade head on the test DB before suite, truncate between tests.
- Mocks: DVC subprocess/S3, MLflow client, Argo HTTP, artifact storage boundaries.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure app imports work
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by directory so `pytest -m 'unit or integration'` works without per-file boilerplate."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path.replace("\\", "/"):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path.replace("\\", "/"):
            item.add_marker(pytest.mark.integration)
        elif "/pipeline/" in path.replace("\\", "/"):
            item.add_marker(pytest.mark.pipeline)

# Patch global DB engine early so security/audit don't hit real Postgres
# This autouse will run for every test and bind app's SessionLocal to test file DB
import app.core.db as _dbmod
import app.core.security as _secmod


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory) -> str:
    env_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL_TEST")
    if env_url:
        return env_url
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    return f"sqlite+pysqlite:///{db_file}"


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    engine = create_engine(
        test_db_url,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False} if test_db_url.startswith("sqlite") else {},
    )
    if test_db_url.startswith("sqlite"):
        import app.core.models  # noqa: F401
        from app.core.db import Base

        Base.metadata.create_all(engine)
    else:
        os.environ["DATABASE_URL"] = test_db_url
        from alembic.config import Config

        from alembic import command

        cfg = Config(str(REPO / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", test_db_url)
        command.upgrade(cfg, "head")
    yield engine
    engine.dispose()
    # reset patched globals after session
    try:
        _dbmod.engine.dispose()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _patch_global_db(test_engine):
    """Ensure all app code (including security) uses test_engine, not real Postgres."""
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    # Monkeypatch module-level SessionLocal/engine
    # Use setattr directly (monkeypatch fixture not available at this scope for session)
    _dbmod.engine = test_engine
    _dbmod.SessionLocal = TestingSession
    _secmod.SessionLocal = TestingSession
    yield
    # no revert needed — next test will re-patch


@pytest.fixture()
def db(test_engine):
    """Per-test session with truncate between tests. Shares test_engine file DB."""
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    session = TestingSession()
    yield session
    # Truncate app tables between tests (FK order: children first)
    try:
        for tbl in ["audit_log", "drift_checks", "predictions", "training_runs", "feature_sets", "dataset_versions", "datasets", "alembic_version"]:
            try:
                # Keep alembic_version for sqlite file-based approach
                if tbl == "alembic_version":
                    continue
                session.execute(text(f"DELETE FROM {tbl}"))
            except Exception:
                session.rollback()
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def client(db, monkeypatch, test_engine):
    """FastAPI TestClient with get_db overridden to a fresh session per request."""
    from app.core.db import get_db
    from app.main import app

    # Each request gets its own session bound to test_engine (isolation)
    def _get_db_override():
        TestingSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
        sess = TestingSession()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    app.dependency_overrides[get_db] = _get_db_override

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {"Contents": []}
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"a,b\n1,2\n")}

    with patch("app.data.dvc_io.s3_client", return_value=mock_s3):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def api_key(monkeypatch):
    # Override settings.api_key for tests
    from app.core.config import settings

    monkeypatch.setattr(settings, "api_key", "test-key-123")
    return "test-key-123"


@pytest.fixture()
def temp_repo(tmp_path, monkeypatch):
    """Temporary repo root for DVC tests — prevents writing to real working tree."""
    # app/data/dvc_io.REPO is Path(__file__).resolve().parents[2]
    # We monkeypatch that location via env and module reload if needed.
    # Simpler: patch dvc_io.REPO and dvc_io._run
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / ".dvc").mkdir()
    (fake_repo / "data" / "raw" / "house-prices").mkdir(parents=True)
    monkeypatch.setenv("PYTEST_TMP_REPO", str(fake_repo))
    return fake_repo


@pytest.fixture()
def auth_header(api_key):
    return {"Authorization": f"Bearer {api_key}"}
