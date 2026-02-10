"""Smoke tests for Pukara gateway.

Uses the in-memory backend — tests the HTTP layer without ArangoDB.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from yanantin.apacheta.backends.memory import InMemoryBackend
from yanantin.apacheta.interface.abstract import ApachetaInterface
from yanantin.apacheta.interface.errors import (
    AccessDeniedError,
    ApachetaError,
    ImmutabilityError,
    InterfaceVersionError,
    NotFoundError,
)
from yanantin.apacheta.models.composition import (
    CompositionEdge,
    RelationType,
)
from yanantin.apacheta.models.tensor import TensorRecord

from pukara.auth import make_api_key_checker
from pukara.decoder import DecoderRing
from pukara.routes import meta, query, read, store


def _make_test_app() -> FastAPI:
    """Create a test app with in-memory backend and exception handlers."""
    app = FastAPI()
    backend = InMemoryBackend()
    app.state.backend = backend
    app.state.decoder = DecoderRing()

    @app.exception_handler(ImmutabilityError)
    async def immutability_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(InterfaceVersionError)
    async def version_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ApachetaError)
    async def apacheta_handler(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    check_key = make_api_key_checker("")  # no auth
    app.include_router(meta.router, dependencies=[Depends(check_key)])
    app.include_router(store.router, dependencies=[Depends(check_key)])
    app.include_router(read.router, dependencies=[Depends(check_key)])
    app.include_router(query.router, dependencies=[Depends(check_key)])
    return app


@pytest.fixture
def client():
    app = _make_test_app()
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["service"] == "pukara"

    def test_version(self, client):
        r = client.get("/api/v1/version")
        assert r.status_code == 200
        assert r.json()["gateway"] == "0.1.0"
        assert r.json()["interface"] == "v1"

    def test_counts_empty(self, client):
        r = client.get("/api/v1/counts")
        assert r.status_code == 200
        data = r.json()
        assert data["tensors"] == 0
        assert data["edges"] == 0


class TestStoreTensor:
    def test_store_and_retrieve(self, client):
        tensor = TensorRecord(preamble="Test tensor via gateway")
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201
        tensor_id = r.json()["id"]

        r = client.get(f"/api/v1/tensors/{tensor_id}")
        assert r.status_code == 200
        assert r.json()["preamble"] == "Test tensor via gateway"

    def test_immutability(self, client):
        tensor = TensorRecord(preamble="Immutable")
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201

        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 409

    def test_not_found(self, client):
        r = client.get(f"/api/v1/tensors/{uuid4()}")
        assert r.status_code == 404

    def test_list_tensors(self, client):
        t1 = TensorRecord(preamble="First")
        t2 = TensorRecord(preamble="Second")
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        r = client.get("/api/v1/tensors")
        assert r.status_code == 200
        assert len(r.json()) == 2


class TestStoreComposition:
    def test_store_composition_edge(self, client):
        edge = CompositionEdge(
            from_tensor=uuid4(),
            to_tensor=uuid4(),
            relation_type=RelationType.COMPOSES_WITH,
        )
        r = client.post(
            "/api/v1/composition-edges",
            json=edge.model_dump(mode="json"),
        )
        assert r.status_code == 201


class TestQueries:
    def test_project_state(self, client):
        r = client.get("/api/v1/queries/project-state")
        assert r.status_code == 200
        assert r.json()["tensor_count"] == 0

    def test_open_questions(self, client):
        r = client.get("/api/v1/queries/open-questions")
        assert r.status_code == 200
        assert r.json() == []

    def test_loss_patterns(self, client):
        r = client.get("/api/v1/queries/loss-patterns")
        assert r.status_code == 200
        assert r.json() == []

    def test_claims_about(self, client):
        r = client.get("/api/v1/queries/claims-about?topic=testing")
        assert r.status_code == 200

    def test_tensors_for_budget(self, client):
        r = client.get("/api/v1/queries/tensors-for-budget?budget=1000")
        assert r.status_code == 200


class TestAuth:
    def test_no_key_required_in_dev_mode(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_key_required_when_configured(self):
        app = FastAPI()
        app.state.backend = InMemoryBackend()
        app.state.decoder = DecoderRing()
        check_key = make_api_key_checker("secret-key-123")
        app.include_router(meta.router, dependencies=[Depends(check_key)])

        c = TestClient(app)

        r = c.get("/api/v1/health")
        assert r.status_code == 401

        r = c.get("/api/v1/health", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

        r = c.get("/api/v1/health", headers={"X-API-Key": "secret-key-123"})
        assert r.status_code == 200
