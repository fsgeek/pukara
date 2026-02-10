"""Pukara — FastAPI gateway application.

The fortress between agents and the tensor database.
Agents can only reach ArangoDB through these endpoints.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from yanantin.apacheta.backends.arango import ArangoDBBackend
from yanantin.apacheta.interface.errors import (
    AccessDeniedError,
    ApachetaError,
    ImmutabilityError,
    InterfaceVersionError,
    NotFoundError,
)

from pukara.auth import make_api_key_checker
from pukara.config import PukaraConfig, load_config
from pukara.decoder import DecoderRing
from pukara.routes import meta, query, read, store

logger = logging.getLogger("pukara")
audit_logger = logging.getLogger("pukara.audit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to ArangoDB. Shutdown: close connection."""
    config: PukaraConfig = app.state.config
    logger.info(
        "Connecting to ArangoDB at %s (database: %s)",
        config.arango_host,
        config.arango_db,
    )
    backend = ArangoDBBackend(
        host=config.arango_host,
        db_name=config.arango_db,
        username=config.arango_user,
        password=config.arango_password,
    )
    app.state.backend = backend
    app.state.decoder = DecoderRing()
    logger.info("Pukara gateway ready")
    yield
    backend.close()
    logger.info("Pukara gateway shut down")


def create_app(config: PukaraConfig | None = None) -> FastAPI:
    """Application factory."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="Pukara",
        description="Fortress gateway — boundary defense for Apacheta tensor database",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config

    check_key = make_api_key_checker(config.api_key)

    # ── Exception handlers ────────────────────────────────────

    @app.exception_handler(ImmutabilityError)
    async def immutability_handler(request: Request, exc: ImmutabilityError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(request: Request, exc: AccessDeniedError):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(InterfaceVersionError)
    async def version_handler(request: Request, exc: InterfaceVersionError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ApachetaError)
    async def apacheta_handler(request: Request, exc: ApachetaError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # ── Audit middleware ──────────────────────────────────────

    @app.middleware("http")
    async def audit_log(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        audit_logger.info(
            "%s %s %d %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # ── Routers ───────────────────────────────────────────────

    app.include_router(meta.router, dependencies=[Depends(check_key)])
    app.include_router(store.router, dependencies=[Depends(check_key)])
    app.include_router(read.router, dependencies=[Depends(check_key)])
    app.include_router(query.router, dependencies=[Depends(check_key)])

    return app
