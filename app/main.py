from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.config import settings
from .data.routes import router as data_router
from .monitoring.routes import router as monitoring_router
from .registry.routes import router as registry_router
from .serving.routes import router as serving_router
from .training.routes import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema is owned by Alembic — do not create_all here.
    # See alembic/env.py and docker/api-entrypoint.sh (alembic upgrade head).
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


app.include_router(data_router)
app.include_router(registry_router)
app.include_router(training_router)
app.include_router(serving_router)
app.include_router(monitoring_router)
