"""Independent tests for Pukara FastAPI gateway.

Written by the test author — NOT the code author. These tests verify
the HTTP layer, exception handling, authentication, edge cases, and
full roundtrip behavior using realistic data.

Uses InMemoryBackend: no ArangoDB dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from yanantin.apacheta.backends.memory import InMemoryBackend
from yanantin.apacheta.interface.errors import (
    AccessDeniedError,
    ApachetaError,
    ImmutabilityError,
    InterfaceVersionError,
    NotFoundError,
)
from yanantin.apacheta.models.composition import (
    BootstrapRecord,
    CompositionEdge,
    CorrectionRecord,
    DissentRecord,
    NegationRecord,
    RelationType,
    SchemaEvolutionRecord,
)
from yanantin.apacheta.models.entities import EntityResolution
from yanantin.apacheta.models.epistemics import (
    DeclaredLoss,
    DisagreementType,
    EpistemicMetadata,
    LossCategory,
    RepresentationType,
)
from yanantin.apacheta.models.provenance import ProvenanceEnvelope, SourceIdentifier
from yanantin.apacheta.models.tensor import KeyClaim, StrandRecord, TensorRecord

from pukara.auth import make_api_key_checker
from pukara.decoder import DecoderRing
from pukara.routes import meta, query, read, store


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_test_app(api_key: str = "") -> FastAPI:
    """Build a test app with in-memory backend and all exception handlers."""
    app = FastAPI()
    app.state.backend = InMemoryBackend()
    app.state.decoder = DecoderRing()

    @app.exception_handler(ImmutabilityError)
    async def _h_immutability(req, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def _h_not_found(req, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AccessDeniedError)
    async def _h_access_denied(req, exc):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(InterfaceVersionError)
    async def _h_version(req, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ApachetaError)
    async def _h_apacheta(req, exc):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    check_key = make_api_key_checker(api_key)
    app.include_router(meta.router, dependencies=[Depends(check_key)])
    app.include_router(store.router, dependencies=[Depends(check_key)])
    app.include_router(read.router, dependencies=[Depends(check_key)])
    app.include_router(query.router, dependencies=[Depends(check_key)])
    return app


# ---------------------------------------------------------------------------
# Realistic data builders
# ---------------------------------------------------------------------------


def _make_provenance(
    *,
    model_family: str = "claude-opus",
    instance_id: str = "test-instance-42",
    budget: float = 128000.0,
    predecessors: tuple[UUID, ...] = (),
) -> ProvenanceEnvelope:
    return ProvenanceEnvelope(
        source=SourceIdentifier(
            version="v1",
            description="test provenance",
        ),
        timestamp=datetime.now(timezone.utc),
        author_model_family=model_family,
        author_instance_id=instance_id,
        context_budget_at_write=budget,
        predecessors_in_scope=predecessors,
        interface_version="v1",
    )


def _make_epistemic(
    truth: float = 0.8,
    indeterminacy: float = 0.1,
    falsity: float = 0.05,
) -> EpistemicMetadata:
    return EpistemicMetadata(
        representation_type=RepresentationType.SCALAR,
        truth=truth,
        indeterminacy=indeterminacy,
        falsity=falsity,
    )


def _make_claim(text: str, *, truth: float = 0.9, indet: float = 0.05) -> KeyClaim:
    return KeyClaim(
        text=text,
        epistemic=_make_epistemic(truth=truth, indeterminacy=indet),
        evidence_refs=("doi:10.1234/test", "arxiv:2301.00000"),
    )


def _make_strand(
    index: int,
    title: str,
    *,
    topics: tuple[str, ...] = (),
    claims: tuple[KeyClaim, ...] = (),
    content: str = "",
) -> StrandRecord:
    return StrandRecord(
        strand_index=index,
        title=title,
        content=content or f"Content for strand {index}: {title}",
        topics=topics,
        key_claims=claims,
        epistemic=_make_epistemic(),
    )


def _make_rich_tensor(
    *,
    preamble: str = "A realistic tensor for testing",
    model_family: str = "claude-opus",
    lineage_tags: tuple[str, ...] = ("T0",),
    strands: tuple[StrandRecord, ...] | None = None,
    open_questions: tuple[str, ...] = (),
    declared_losses: tuple[DeclaredLoss, ...] = (),
) -> TensorRecord:
    if strands is None:
        strands = (
            _make_strand(
                0,
                "Epistemic Foundations",
                topics=("epistemics", "neutrosophic-logic"),
                claims=(
                    _make_claim("Neutrosophic T/I/F values are independent"),
                    _make_claim("Context pressure causes information loss"),
                ),
            ),
            _make_strand(
                1,
                "Architecture Decisions",
                topics=("architecture", "design"),
                claims=(
                    _make_claim("Immutability is a structural constraint"),
                    _make_claim("The decoder ring separates internal and external UUIDs"),
                ),
                content="Architecture notes for the Apacheta system.",
            ),
        )
    return TensorRecord(
        provenance=_make_provenance(model_family=model_family),
        preamble=preamble,
        strands=strands,
        closing="End of tensor",
        instructions_for_next="Continue the lineage",
        narrative_body="Full markdown body of the tensor\n\n## Section\nNarrative here.",
        lineage_tags=lineage_tags,
        composition_equation="T0 + T1 -> T2",
        declared_losses=declared_losses,
        epistemic=_make_epistemic(),
        open_questions=open_questions,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Fresh test app per test."""
    return _make_test_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def authed_app():
    """App that requires an API key."""
    return _make_test_app(api_key="fortress-key-Pukara-2025")


@pytest.fixture
def authed_client(authed_app):
    return TestClient(authed_app)


# ---------------------------------------------------------------------------
# 1. Store endpoint tests — all 8 record types
# ---------------------------------------------------------------------------


class TestStoreTensor:
    """POST /api/v1/tensors with realistic tensor data."""

    def test_store_rich_tensor_returns_201_and_uuid(self, client):
        tensor = _make_rich_tensor()
        payload = tensor.model_dump(mode="json")
        r = client.post("/api/v1/tensors", json=payload)
        assert r.status_code == 201
        body = r.json()
        assert "id" in body
        # Returned ID must parse as a valid UUID
        returned = UUID(body["id"])
        assert returned == tensor.id

    def test_store_tensor_with_declared_losses(self, client):
        losses = (
            DeclaredLoss(
                what_was_lost="Emotional texture of original conversation",
                why="Context window pressure forced compression",
                category=LossCategory.CONTEXT_PRESSURE,
            ),
            DeclaredLoss(
                what_was_lost="Alternative framework considered but discarded",
                why="Authorial choice to focus on primary thread",
                category=LossCategory.AUTHORIAL_CHOICE,
            ),
        )
        tensor = _make_rich_tensor(
            declared_losses=losses,
            open_questions=(
                "Can epistemic metadata survive multiple compressions?",
                "Does the decoder ring introduce UUID collision risks?",
            ),
        )
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201

    def test_store_tensor_with_many_strands(self, client):
        strands = tuple(
            _make_strand(
                i,
                f"Strand {i}",
                topics=(f"topic-{i}",),
                claims=(_make_claim(f"Claim in strand {i}"),),
            )
            for i in range(10)
        )
        tensor = _make_rich_tensor(strands=strands)
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201

    def test_store_minimal_tensor(self, client):
        """A tensor with only defaults — bare minimum."""
        tensor = TensorRecord()
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == tensor.id

    def test_store_tensor_immutability_409(self, client):
        tensor = _make_rich_tensor(preamble="Immutable tensor")
        payload = tensor.model_dump(mode="json")
        r1 = client.post("/api/v1/tensors", json=payload)
        assert r1.status_code == 201

        r2 = client.post("/api/v1/tensors", json=payload)
        assert r2.status_code == 409
        assert "detail" in r2.json()
        assert "already exists" in r2.json()["detail"].lower() or "immutable" in r2.json()["detail"].lower()

    def test_store_tensor_invalid_json_returns_422(self, client):
        r = client.post(
            "/api/v1/tensors",
            content=b'{"id": "not-a-uuid"}',
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422


class TestStoreCompositionEdge:
    """POST /api/v1/composition-edges"""

    def test_store_composes_with_edge(self, client):
        t1_id, t2_id = uuid4(), uuid4()
        edge = CompositionEdge(
            from_tensor=t1_id,
            to_tensor=t2_id,
            relation_type=RelationType.COMPOSES_WITH,
            ordering=1,
            authored_mapping="T0 strand 2 maps to T1 strand 0",
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/composition-edges", json=edge.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == edge.id

    def test_store_corrects_edge(self, client):
        edge = CompositionEdge(
            from_tensor=uuid4(),
            to_tensor=uuid4(),
            relation_type=RelationType.CORRECTS,
            ordering=0,
        )
        r = client.post("/api/v1/composition-edges", json=edge.model_dump(mode="json"))
        assert r.status_code == 201

    def test_store_all_relation_types(self, client):
        """Every RelationType should be accepted."""
        for rt in RelationType:
            edge = CompositionEdge(
                from_tensor=uuid4(),
                to_tensor=uuid4(),
                relation_type=rt,
            )
            r = client.post("/api/v1/composition-edges", json=edge.model_dump(mode="json"))
            assert r.status_code == 201, f"Failed for relation type {rt.value}"

    def test_composition_edge_immutability(self, client):
        edge = CompositionEdge(
            from_tensor=uuid4(),
            to_tensor=uuid4(),
            relation_type=RelationType.REFINES,
        )
        payload = edge.model_dump(mode="json")
        r1 = client.post("/api/v1/composition-edges", json=payload)
        assert r1.status_code == 201
        r2 = client.post("/api/v1/composition-edges", json=payload)
        assert r2.status_code == 409


class TestStoreCorrection:
    """POST /api/v1/corrections"""

    def test_store_correction_with_claim_target(self, client):
        target_tensor = uuid4()
        target_claim = uuid4()
        correction = CorrectionRecord(
            target_tensor=target_tensor,
            target_strand_index=0,
            target_claim_id=target_claim,
            original_claim="Neutrosophic values must sum to 1.0",
            corrected_claim="Neutrosophic T/I/F values are independent — no summation constraint",
            evidence="See Section 2 of Smarandache (1999)",
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/corrections", json=correction.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == correction.id

    def test_store_correction_without_claim_target(self, client):
        correction = CorrectionRecord(
            target_tensor=uuid4(),
            original_claim="Old statement",
            corrected_claim="Updated statement",
        )
        r = client.post("/api/v1/corrections", json=correction.model_dump(mode="json"))
        assert r.status_code == 201

    def test_correction_immutability(self, client):
        correction = CorrectionRecord(
            target_tensor=uuid4(),
            original_claim="X",
            corrected_claim="Y",
        )
        payload = correction.model_dump(mode="json")
        assert client.post("/api/v1/corrections", json=payload).status_code == 201
        assert client.post("/api/v1/corrections", json=payload).status_code == 409


class TestStoreDissent:
    """POST /api/v1/dissents"""

    def test_store_dissent_with_reasoning(self, client):
        dissent = DissentRecord(
            target_tensor=uuid4(),
            target_claim_id=uuid4(),
            alternative_framework="Bayesian probability is more appropriate here than neutrosophic logic",
            reasoning=(
                "The claimed independence of T/I/F is useful for philosophical discourse "
                "but operationally misleading when agents need actionable confidence scores."
            ),
            provenance=_make_provenance(model_family="gpt-4o"),
        )
        r = client.post("/api/v1/dissents", json=dissent.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == dissent.id

    def test_dissent_immutability(self, client):
        dissent = DissentRecord(
            target_tensor=uuid4(),
            alternative_framework="alt",
            reasoning="reason",
        )
        payload = dissent.model_dump(mode="json")
        assert client.post("/api/v1/dissents", json=payload).status_code == 201
        assert client.post("/api/v1/dissents", json=payload).status_code == 409


class TestStoreNegation:
    """POST /api/v1/negations"""

    def test_store_negation(self, client):
        negation = NegationRecord(
            tensor_a=uuid4(),
            tensor_b=uuid4(),
            reasoning=(
                "T3 assumes a collaborative framework; T5 assumes an adversarial one. "
                "Composing them would produce incoherent operational principles."
            ),
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/negations", json=negation.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == negation.id

    def test_negation_immutability(self, client):
        negation = NegationRecord(
            tensor_a=uuid4(),
            tensor_b=uuid4(),
            reasoning="They don't compose",
        )
        payload = negation.model_dump(mode="json")
        assert client.post("/api/v1/negations", json=payload).status_code == 201
        assert client.post("/api/v1/negations", json=payload).status_code == 409


class TestStoreBootstrap:
    """POST /api/v1/bootstraps"""

    def test_store_bootstrap_with_selections(self, client):
        t1, t2, t3 = uuid4(), uuid4(), uuid4()
        bootstrap = BootstrapRecord(
            instance_id="claude-opus-instance-9f3a",
            context_budget=128000.0,
            task="Continue the tensor sequence from T7",
            tensors_selected=(t1, t2, t3),
            strands_selected=(0, 1, 3),
            what_was_omitted="T4 and T5 omitted due to context budget. T6 strand 2 dropped (tangential).",
            provenance=_make_provenance(budget=128000.0),
        )
        r = client.post("/api/v1/bootstraps", json=bootstrap.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == bootstrap.id

    def test_store_minimal_bootstrap(self, client):
        bootstrap = BootstrapRecord(
            instance_id="test",
            context_budget=0.0,
        )
        r = client.post("/api/v1/bootstraps", json=bootstrap.model_dump(mode="json"))
        assert r.status_code == 201

    def test_bootstrap_immutability(self, client):
        bootstrap = BootstrapRecord(
            instance_id="test-immutable",
            context_budget=1000.0,
        )
        payload = bootstrap.model_dump(mode="json")
        assert client.post("/api/v1/bootstraps", json=payload).status_code == 201
        assert client.post("/api/v1/bootstraps", json=payload).status_code == 409


class TestStoreEvolution:
    """POST /api/v1/evolutions"""

    def test_store_schema_evolution(self, client):
        evolution = SchemaEvolutionRecord(
            from_version="v1",
            to_version="v2",
            fields_added=("functional_spec", "scope_boundaries"),
            fields_removed=(),
            migration_notes="Added functional epistemic representation type. No breaking changes.",
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/evolutions", json=evolution.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == evolution.id

    def test_evolution_immutability(self, client):
        evolution = SchemaEvolutionRecord(
            from_version="v0",
            to_version="v1",
        )
        payload = evolution.model_dump(mode="json")
        assert client.post("/api/v1/evolutions", json=payload).status_code == 201
        assert client.post("/api/v1/evolutions", json=payload).status_code == 409


class TestStoreEntity:
    """POST /api/v1/entities"""

    def test_store_entity_resolution(self, client):
        entity_uuid = uuid4()
        entity = EntityResolution(
            entity_uuid=entity_uuid,
            identity_type="model_instance",
            identity_data={
                "model_family": "claude-opus",
                "instance_id": "claude-opus-4-20250514",
                "provider": "anthropic",
            },
            redacted=False,
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/entities", json=entity.model_dump(mode="json"))
        assert r.status_code == 201
        assert UUID(r.json()["id"]) == entity.id

    def test_store_redacted_entity(self, client):
        entity = EntityResolution(
            entity_uuid=uuid4(),
            identity_type="human",
            identity_data={"name": "[REDACTED]"},
            redacted=True,
        )
        r = client.post("/api/v1/entities", json=entity.model_dump(mode="json"))
        assert r.status_code == 201

    def test_entity_immutability(self, client):
        entity = EntityResolution(
            entity_uuid=uuid4(),
            identity_type="test",
            identity_data={},
        )
        payload = entity.model_dump(mode="json")
        assert client.post("/api/v1/entities", json=payload).status_code == 201
        assert client.post("/api/v1/entities", json=payload).status_code == 409


# ---------------------------------------------------------------------------
# 2. Read endpoint tests — all 4 endpoints including error cases
# ---------------------------------------------------------------------------


class TestReadTensors:
    """GET /api/v1/tensors and GET /api/v1/tensors/{id}"""

    def test_list_tensors_empty(self, client):
        r = client.get("/api/v1/tensors")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_tensors_after_storing_multiple(self, client):
        for i in range(3):
            t = _make_rich_tensor(preamble=f"Tensor {i}", model_family="claude-opus")
            client.post("/api/v1/tensors", json=t.model_dump(mode="json"))

        r = client.get("/api/v1/tensors")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3
        preambles = {d["preamble"] for d in data}
        assert preambles == {"Tensor 0", "Tensor 1", "Tensor 2"}

    def test_get_tensor_by_id(self, client):
        tensor = _make_rich_tensor(
            preamble="Specific tensor for retrieval",
            lineage_tags=("T0", "T1"),
        )
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}")
        assert r.status_code == 200
        data = r.json()
        assert data["preamble"] == "Specific tensor for retrieval"
        assert "T0" in data["lineage_tags"]
        assert "T1" in data["lineage_tags"]
        assert len(data["strands"]) == 2
        assert data["closing"] == "End of tensor"
        assert data["narrative_body"].startswith("Full markdown body")

    def test_get_tensor_not_found_404(self, client):
        missing_id = uuid4()
        r = client.get(f"/api/v1/tensors/{missing_id}")
        assert r.status_code == 404
        assert "detail" in r.json()

    def test_get_tensor_preserves_epistemic_metadata(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}")
        data = r.json()
        # Check top-level epistemic
        assert data["epistemic"]["truth"] == pytest.approx(0.8)
        assert data["epistemic"]["indeterminacy"] == pytest.approx(0.1)
        assert data["epistemic"]["falsity"] == pytest.approx(0.05)
        # Check strand-level epistemic
        strand_0 = data["strands"][0]
        assert strand_0["epistemic"]["representation_type"] == "scalar"

    def test_get_tensor_preserves_provenance(self, client):
        tensor = _make_rich_tensor(model_family="mistral-large")
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}")
        data = r.json()
        assert data["provenance"]["author_model_family"] == "mistral-large"
        assert data["provenance"]["author_instance_id"] == "test-instance-42"
        assert data["provenance"]["interface_version"] == "v1"


class TestReadStrands:
    """GET /api/v1/tensors/{tensor_id}/strands/{strand_index}"""

    def test_get_strand_by_index(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}/strands/0")
        assert r.status_code == 200
        data = r.json()
        # The response is a TensorRecord projection with only the one strand
        assert len(data["strands"]) == 1
        assert data["strands"][0]["strand_index"] == 0
        assert data["strands"][0]["title"] == "Epistemic Foundations"
        # The tensor-level fields should still be present
        assert data["id"] == str(tensor.id)
        assert data["preamble"] == "A realistic tensor for testing"

    def test_get_second_strand(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}/strands/1")
        assert r.status_code == 200
        data = r.json()
        assert len(data["strands"]) == 1
        assert data["strands"][0]["title"] == "Architecture Decisions"

    def test_get_strand_not_found_tensor_missing(self, client):
        r = client.get(f"/api/v1/tensors/{uuid4()}/strands/0")
        assert r.status_code == 404

    def test_get_strand_not_found_bad_index(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}/strands/999")
        assert r.status_code == 404
        assert "detail" in r.json()

    def test_strand_preserves_key_claims(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/tensors/{tensor.id}/strands/0")
        data = r.json()
        claims = data["strands"][0]["key_claims"]
        assert len(claims) == 2
        claim_texts = {c["text"] for c in claims}
        assert "Neutrosophic T/I/F values are independent" in claim_texts
        assert "Context pressure causes information loss" in claim_texts


class TestReadEntities:
    """GET /api/v1/entities/{entity_id}"""

    def test_get_entity_by_id(self, client):
        entity = EntityResolution(
            entity_uuid=uuid4(),
            identity_type="model_instance",
            identity_data={"model": "claude-opus", "version": "4"},
        )
        client.post("/api/v1/entities", json=entity.model_dump(mode="json"))

        r = client.get(f"/api/v1/entities/{entity.id}")
        assert r.status_code == 200
        data = r.json()
        assert data["identity_type"] == "model_instance"
        assert data["identity_data"]["model"] == "claude-opus"

    def test_get_entity_not_found(self, client):
        r = client.get(f"/api/v1/entities/{uuid4()}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. Query endpoints with actual stored data
# ---------------------------------------------------------------------------


class TestQueryWithData:
    """Store tensors first, then exercise query endpoints."""

    @pytest.fixture
    def populated_client(self, client):
        """Client with 3 tensors, 2 edges, 1 correction, 1 dissent, 1 negation stored."""
        # Tensor 0 — claude-opus, lineage T0
        self.t0 = _make_rich_tensor(
            preamble="T0: The origin tensor",
            model_family="claude-opus",
            lineage_tags=("T0",),
            open_questions=(
                "How do we measure compression loss?",
                "Is the decoder ring necessary for v1?",
            ),
            declared_losses=(
                DeclaredLoss(
                    what_was_lost="Emotional context",
                    why="Compression",
                    category=LossCategory.CONTEXT_PRESSURE,
                ),
            ),
        )
        # Tensor 1 — gpt-4o, lineage T0 (same lineage as T0)
        self.t1 = _make_rich_tensor(
            preamble="T1: Cross-model response",
            model_family="gpt-4o",
            lineage_tags=("T0", "T1"),
            strands=(
                _make_strand(
                    0,
                    "Error Patterns",
                    topics=("error", "failure", "anti-pattern"),
                    claims=(
                        _make_claim("Mock services hide real failures", truth=0.95),
                        _make_claim(
                            "This signal is unreliable",
                            truth=0.3,
                            indet=0.7,
                        ),
                    ),
                ),
            ),
            declared_losses=(
                DeclaredLoss(
                    what_was_lost="Traversal context",
                    why="Chose depth over breadth",
                    category=LossCategory.TRAVERSAL_BIAS,
                ),
            ),
        )
        # Tensor 2 — different lineage
        self.t2 = _make_rich_tensor(
            preamble="T2: Independent branch",
            model_family="claude-opus",
            lineage_tags=("branch-A",),
        )

        for t in (self.t0, self.t1, self.t2):
            r = client.post("/api/v1/tensors", json=t.model_dump(mode="json"))
            assert r.status_code == 201

        # Composition edges
        self.edge_01 = CompositionEdge(
            from_tensor=self.t0.id,
            to_tensor=self.t1.id,
            relation_type=RelationType.COMPOSES_WITH,
            ordering=1,
            authored_mapping="T0 strand 0 -> T1 strand 0",
            provenance=_make_provenance(),
        )
        self.edge_02 = CompositionEdge(
            from_tensor=self.t0.id,
            to_tensor=self.t2.id,
            relation_type=RelationType.BRANCHES_FROM,
            ordering=2,
        )
        for e in (self.edge_01, self.edge_02):
            r = client.post("/api/v1/composition-edges", json=e.model_dump(mode="json"))
            assert r.status_code == 201

        # Correction targeting a claim in t0
        t0_claim_id = self.t0.strands[0].key_claims[0].claim_id
        self.correction = CorrectionRecord(
            target_tensor=self.t0.id,
            target_strand_index=0,
            target_claim_id=t0_claim_id,
            original_claim="Neutrosophic T/I/F values are independent",
            corrected_claim="Neutrosophic T/I/F values are independent but should be calibrated",
            evidence="Practical experience with uncalibrated values",
            provenance=_make_provenance(),
        )
        r = client.post("/api/v1/corrections", json=self.correction.model_dump(mode="json"))
        assert r.status_code == 201

        # Dissent against T1
        self.dissent = DissentRecord(
            target_tensor=self.t1.id,
            alternative_framework="Not all mock services are harmful",
            reasoning="Integration tests can benefit from controlled mocks",
            provenance=_make_provenance(model_family="llama-3"),
        )
        r = client.post("/api/v1/dissents", json=self.dissent.model_dump(mode="json"))
        assert r.status_code == 201

        # Negation between T1 and T2
        self.negation = NegationRecord(
            tensor_a=self.t1.id,
            tensor_b=self.t2.id,
            reasoning="Incompatible frameworks",
        )
        r = client.post("/api/v1/negations", json=self.negation.model_dump(mode="json"))
        assert r.status_code == 201

        # Entity
        self.entity_uuid = uuid4()
        entity = EntityResolution(
            entity_uuid=self.entity_uuid,
            identity_type="model_instance",
            identity_data={"model": "claude-opus"},
        )
        r = client.post("/api/v1/entities", json=entity.model_dump(mode="json"))
        assert r.status_code == 201
        self.entity = entity

        return client

    def test_counts_after_population(self, populated_client):
        r = populated_client.get("/api/v1/counts")
        assert r.status_code == 200
        data = r.json()
        assert data["tensors"] == 3
        assert data["edges"] == 2
        assert data["corrections"] == 1
        assert data["dissents"] == 1
        assert data["negations"] == 1
        assert data["entities"] == 1

    def test_project_state(self, populated_client):
        r = populated_client.get("/api/v1/queries/project-state")
        assert r.status_code == 200
        data = r.json()
        assert data["tensor_count"] == 3
        assert "T0" in data["lineage_tags"]
        assert "T1" in data["lineage_tags"]
        assert "branch-A" in data["lineage_tags"]
        assert "claude-opus" in data["model_families"]
        assert "gpt-4o" in data["model_families"]

    def test_tensors_for_budget(self, populated_client):
        r = populated_client.get("/api/v1/queries/tensors-for-budget?budget=50000")
        assert r.status_code == 200
        data = r.json()
        # InMemoryBackend returns all tensors regardless of budget
        assert len(data) == 3

    def test_operational_principles(self, populated_client):
        r = populated_client.get("/api/v1/queries/operational-principles")
        assert r.status_code == 200
        data = r.json()
        # All key_claims across all strands
        assert isinstance(data, list)
        assert len(data) > 0

    def test_claims_about_topic(self, populated_client):
        r = populated_client.get("/api/v1/queries/claims-about?topic=neutrosophic")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        # Should find claims from the "Epistemic Foundations" strand
        claim_texts = [c["claim"] for c in data]
        assert any("Neutrosophic" in t or "neutrosophic" in t for t in claim_texts)

    def test_claims_about_by_strand_topic(self, populated_client):
        """Search by a strand topic rather than claim text."""
        r = populated_client.get("/api/v1/queries/claims-about?topic=error")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        # Should find claims from T1's "Error Patterns" strand
        assert any("Mock services" in c["claim"] for c in data)

    def test_claims_about_nonexistent_topic(self, populated_client):
        r = populated_client.get("/api/v1/queries/claims-about?topic=xyzzy-nonexistent-99")
        assert r.status_code == 200
        assert r.json() == []

    def test_correction_chain(self, populated_client):
        claim_id = self.t0.strands[0].key_claims[0].claim_id
        r = populated_client.get(f"/api/v1/queries/correction-chain/{claim_id}")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["corrected_claim"] == "Neutrosophic T/I/F values are independent but should be calibrated"

    def test_correction_chain_empty(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/correction-chain/{uuid4()}")
        assert r.status_code == 200
        assert r.json() == []

    def test_epistemic_status_with_corrections(self, populated_client):
        claim_id = self.t0.strands[0].key_claims[0].claim_id
        r = populated_client.get(f"/api/v1/queries/epistemic-status/{claim_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["correction_count"] == 1
        assert "calibrated" in data["current_claim"]
        assert data["original_claim"] == "Neutrosophic T/I/F values are independent"

    def test_epistemic_status_no_corrections(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/epistemic-status/{uuid4()}")
        assert r.status_code == 200
        data = r.json()
        assert data["correction_count"] == 0
        assert data["current_claim"] is None

    def test_disagreements(self, populated_client):
        r = populated_client.get("/api/v1/queries/disagreements")
        assert r.status_code == 200
        data = r.json()
        types_present = {d["type"] for d in data}
        assert "dissent" in types_present
        assert "negation" in types_present
        assert "correction" in types_present
        assert len(data) == 3  # 1 dissent + 1 negation + 1 correction

    def test_composition_graph(self, populated_client):
        r = populated_client.get("/api/v1/queries/composition-graph")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        relation_types = {e["relation_type"] for e in data}
        assert "composes_with" in relation_types
        assert "branches_from" in relation_types

    def test_lineage(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/lineage/{self.t0.id}")
        assert r.status_code == 200
        data = r.json()
        # T0 has lineage_tag "T0". T1 also has "T0". So both should appear.
        assert len(data) >= 2
        ids = {d["id"] for d in data}
        assert str(self.t0.id) in ids
        assert str(self.t1.id) in ids

    def test_lineage_not_found(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/lineage/{uuid4()}")
        assert r.status_code == 404

    def test_bridges(self, populated_client):
        r = populated_client.get("/api/v1/queries/bridges")
        assert r.status_code == 200
        data = r.json()
        # Only edge_01 has authored_mapping
        assert len(data) == 1
        assert data[0]["authored_mapping"] == "T0 strand 0 -> T1 strand 0"

    def test_error_classes(self, populated_client):
        r = populated_client.get("/api/v1/queries/error-classes")
        assert r.status_code == 200
        data = r.json()
        # T1 has strand with topics ("error", "failure", "anti-pattern")
        assert len(data) > 0
        topics = {d["topic"] for d in data}
        assert "error" in topics or "failure" in topics

    def test_open_questions(self, populated_client):
        r = populated_client.get("/api/v1/queries/open-questions")
        assert r.status_code == 200
        data = r.json()
        assert "How do we measure compression loss?" in data
        assert "Is the decoder ring necessary for v1?" in data

    def test_unreliable_signals(self, populated_client):
        r = populated_client.get("/api/v1/queries/unreliable-signals")
        assert r.status_code == 200
        data = r.json()
        # T1 strand 0 has a claim with indeterminacy 0.7 > 0.5
        assert len(data) >= 1
        assert any(d["indeterminacy"] > 0.5 for d in data)
        assert any("unreliable" in d["claim"].lower() for d in data)

    def test_anti_patterns(self, populated_client):
        r = populated_client.get("/api/v1/queries/anti-patterns")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        assert any("anti-pattern" in d["topic"].lower() for d in data)

    def test_authorship(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/authorship/{self.t1.id}")
        assert r.status_code == 200
        data = r.json()
        assert data["author_model_family"] == "gpt-4o"
        assert data["author_instance_id"] == "test-instance-42"
        assert data["context_budget"] == pytest.approx(128000.0)

    def test_authorship_not_found(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/authorship/{uuid4()}")
        assert r.status_code == 404

    def test_cross_model(self, populated_client):
        r = populated_client.get("/api/v1/queries/cross-model")
        assert r.status_code == 200
        data = r.json()
        # We have claude-opus and gpt-4o, so >1 family -> returns all tensors
        assert len(data) == 3

    def test_reading_order(self, populated_client):
        r = populated_client.get("/api/v1/queries/reading-order?tag=T0")
        assert r.status_code == 200
        data = r.json()
        # T0 and T1 both have lineage_tag "T0"
        assert len(data) == 2
        # Should be sorted by timestamp
        if len(data) >= 2:
            ts0 = data[0]["provenance"]["timestamp"]
            ts1 = data[1]["provenance"]["timestamp"]
            assert ts0 <= ts1

    def test_reading_order_no_matches(self, populated_client):
        r = populated_client.get("/api/v1/queries/reading-order?tag=nonexistent")
        assert r.status_code == 200
        assert r.json() == []

    def test_unlearn(self, populated_client):
        r = populated_client.get("/api/v1/queries/unlearn?topic=neutrosophic")
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "neutrosophic"
        assert data["affected_claims"] > 0
        assert len(data["affected_tensors"]) > 0

    def test_losses(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/losses/{self.t0.id}")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["category"] == "context_pressure"
        assert "Emotional" in data[0]["what"]

    def test_losses_not_found(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/losses/{uuid4()}")
        assert r.status_code == 404

    def test_loss_patterns(self, populated_client):
        r = populated_client.get("/api/v1/queries/loss-patterns")
        assert r.status_code == 200
        data = r.json()
        categories = {d["category"] for d in data}
        assert "context_pressure" in categories
        assert "traversal_bias" in categories

    def test_entities_by_uuid(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/entities-by-uuid/{self.entity_uuid}")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["identity_type"] == "model_instance"

    def test_entities_by_uuid_no_matches(self, populated_client):
        r = populated_client.get(f"/api/v1/queries/entities-by-uuid/{uuid4()}")
        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# 4. Exception handler mapping tests
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    """Verify each Apacheta error maps to the correct HTTP status."""

    def test_immutability_error_maps_to_409(self, client):
        tensor = TensorRecord()
        payload = tensor.model_dump(mode="json")
        client.post("/api/v1/tensors", json=payload)
        r = client.post("/api/v1/tensors", json=payload)
        assert r.status_code == 409
        assert "detail" in r.json()

    def test_not_found_error_maps_to_404(self, client):
        r = client.get(f"/api/v1/tensors/{uuid4()}")
        assert r.status_code == 404
        assert "detail" in r.json()

    def test_not_found_detail_contains_uuid(self, client):
        missing = uuid4()
        r = client.get(f"/api/v1/tensors/{missing}")
        assert str(missing) in r.json()["detail"]

    def test_409_on_edge_duplicate(self, client):
        edge = CompositionEdge(
            from_tensor=uuid4(), to_tensor=uuid4(),
            relation_type=RelationType.COMPOSES_WITH,
        )
        payload = edge.model_dump(mode="json")
        client.post("/api/v1/composition-edges", json=payload)
        r = client.post("/api/v1/composition-edges", json=payload)
        assert r.status_code == 409

    def test_409_on_correction_duplicate(self, client):
        c = CorrectionRecord(
            target_tensor=uuid4(),
            original_claim="A", corrected_claim="B",
        )
        payload = c.model_dump(mode="json")
        client.post("/api/v1/corrections", json=payload)
        r = client.post("/api/v1/corrections", json=payload)
        assert r.status_code == 409

    def test_409_on_dissent_duplicate(self, client):
        d = DissentRecord(
            target_tensor=uuid4(),
            alternative_framework="X", reasoning="Y",
        )
        payload = d.model_dump(mode="json")
        client.post("/api/v1/dissents", json=payload)
        r = client.post("/api/v1/dissents", json=payload)
        assert r.status_code == 409

    def test_409_on_negation_duplicate(self, client):
        n = NegationRecord(
            tensor_a=uuid4(), tensor_b=uuid4(), reasoning="Z",
        )
        payload = n.model_dump(mode="json")
        client.post("/api/v1/negations", json=payload)
        r = client.post("/api/v1/negations", json=payload)
        assert r.status_code == 409

    def test_409_on_bootstrap_duplicate(self, client):
        b = BootstrapRecord(instance_id="x", context_budget=1.0)
        payload = b.model_dump(mode="json")
        client.post("/api/v1/bootstraps", json=payload)
        r = client.post("/api/v1/bootstraps", json=payload)
        assert r.status_code == 409

    def test_409_on_evolution_duplicate(self, client):
        e = SchemaEvolutionRecord(from_version="v0", to_version="v1")
        payload = e.model_dump(mode="json")
        client.post("/api/v1/evolutions", json=payload)
        r = client.post("/api/v1/evolutions", json=payload)
        assert r.status_code == 409

    def test_409_on_entity_duplicate(self, client):
        e = EntityResolution(
            entity_uuid=uuid4(), identity_type="test", identity_data={},
        )
        payload = e.model_dump(mode="json")
        client.post("/api/v1/entities", json=payload)
        r = client.post("/api/v1/entities", json=payload)
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# 5. API key enforcement tests
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    """Test API key authentication across all router groups."""

    def test_dev_mode_no_key_allows_all(self, client):
        """Empty API key = dev mode, all requests pass."""
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/version").status_code == 200
        assert client.get("/api/v1/counts").status_code == 200
        assert client.get("/api/v1/tensors").status_code == 200
        assert client.get("/api/v1/queries/project-state").status_code == 200

    def test_required_key_rejects_no_key(self, authed_client):
        r = authed_client.get("/api/v1/health")
        assert r.status_code == 401
        assert "Invalid or missing" in r.json()["detail"]

    def test_required_key_rejects_wrong_key(self, authed_client):
        r = authed_client.get(
            "/api/v1/health",
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_required_key_accepts_correct_key(self, authed_client):
        r = authed_client.get(
            "/api/v1/health",
            headers={"X-API-Key": "fortress-key-Pukara-2025"},
        )
        assert r.status_code == 200

    def test_auth_enforced_on_store_endpoint(self, authed_client):
        tensor = TensorRecord()
        r = authed_client.post(
            "/api/v1/tensors",
            json=tensor.model_dump(mode="json"),
        )
        assert r.status_code == 401

    def test_auth_enforced_on_read_endpoint(self, authed_client):
        r = authed_client.get(f"/api/v1/tensors/{uuid4()}")
        assert r.status_code == 401

    def test_auth_enforced_on_query_endpoint(self, authed_client):
        r = authed_client.get("/api/v1/queries/claims-about?topic=test")
        assert r.status_code == 401

    def test_store_with_correct_key_succeeds(self, authed_client):
        tensor = TensorRecord(preamble="Authenticated store")
        headers = {"X-API-Key": "fortress-key-Pukara-2025"}
        r = authed_client.post(
            "/api/v1/tensors",
            json=tensor.model_dump(mode="json"),
            headers=headers,
        )
        assert r.status_code == 201

    def test_read_with_correct_key_succeeds(self, authed_client):
        headers = {"X-API-Key": "fortress-key-Pukara-2025"}
        tensor = TensorRecord(preamble="Read me")
        authed_client.post(
            "/api/v1/tensors",
            json=tensor.model_dump(mode="json"),
            headers=headers,
        )
        r = authed_client.get(f"/api/v1/tensors/{tensor.id}", headers=headers)
        assert r.status_code == 200

    def test_empty_string_key_header_rejected(self, authed_client):
        """An empty X-API-Key header should still be rejected when a key is configured."""
        r = authed_client.get(
            "/api/v1/health",
            headers={"X-API-Key": ""},
        )
        assert r.status_code == 401

    def test_key_is_case_sensitive(self, authed_client):
        r = authed_client.get(
            "/api/v1/health",
            headers={"X-API-Key": "FORTRESS-KEY-PUKARA-2025"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# 6. Edge cases — invalid paths, missing params, bad UUIDs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge conditions that real clients would hit."""

    def test_invalid_uuid_in_tensor_path(self, client):
        r = client.get("/api/v1/tensors/not-a-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_entity_path(self, client):
        r = client.get("/api/v1/entities/definitely-not-a-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_strand_path(self, client):
        r = client.get("/api/v1/tensors/bad-uuid/strands/0")
        assert r.status_code == 422

    def test_invalid_strand_index_type(self, client):
        """strand_index must be int, not string."""
        tid = uuid4()
        r = client.get(f"/api/v1/tensors/{tid}/strands/abc")
        assert r.status_code == 422

    def test_missing_required_query_param_budget(self, client):
        """budget is required for tensors-for-budget."""
        r = client.get("/api/v1/queries/tensors-for-budget")
        assert r.status_code == 422

    def test_missing_required_query_param_topic(self, client):
        """topic is required for claims-about."""
        r = client.get("/api/v1/queries/claims-about")
        assert r.status_code == 422

    def test_missing_required_query_param_tag(self, client):
        """tag is required for reading-order."""
        r = client.get("/api/v1/queries/reading-order")
        assert r.status_code == 422

    def test_missing_required_query_param_unlearn_topic(self, client):
        """topic is required for unlearn."""
        r = client.get("/api/v1/queries/unlearn")
        assert r.status_code == 422

    def test_invalid_uuid_in_correction_chain_path(self, client):
        r = client.get("/api/v1/queries/correction-chain/not-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_epistemic_status_path(self, client):
        r = client.get("/api/v1/queries/epistemic-status/not-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_lineage_path(self, client):
        r = client.get("/api/v1/queries/lineage/not-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_authorship_path(self, client):
        r = client.get("/api/v1/queries/authorship/not-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_losses_path(self, client):
        r = client.get("/api/v1/queries/losses/not-uuid")
        assert r.status_code == 422

    def test_invalid_uuid_in_entities_by_uuid_path(self, client):
        r = client.get("/api/v1/queries/entities-by-uuid/not-uuid")
        assert r.status_code == 422

    def test_post_empty_body(self, client):
        r = client.post(
            "/api/v1/tensors",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_post_malformed_json(self, client):
        r = client.post(
            "/api/v1/tensors",
            content=b"{broken json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_post_with_extra_fields_rejected(self, client):
        """ApachetaBaseModel uses extra='forbid'."""
        payload = {"preamble": "test", "sneaky_extra_field": "should fail"}
        r = client.post("/api/v1/tensors", json=payload)
        assert r.status_code == 422

    def test_composition_edge_invalid_relation_type(self, client):
        payload = {
            "from_tensor": str(uuid4()),
            "to_tensor": str(uuid4()),
            "relation_type": "invalid_type",
        }
        r = client.post("/api/v1/composition-edges", json=payload)
        assert r.status_code == 422

    def test_negative_strand_index(self, client):
        """Negative strand index should return 404 (not found) since no strand matches."""
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        r = client.get(f"/api/v1/tensors/{tensor.id}/strands/-1")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Full roundtrip: store tensor with strands+claims, then query claims_about
# ---------------------------------------------------------------------------


class TestFullRoundtrip:
    """End-to-end: store rich data, then verify queries return correct results."""

    def test_store_tensor_then_query_claims(self, client):
        """Store a tensor with epistemic claims, then query by topic."""
        claim_a = _make_claim("Memory persistence requires structural redundancy")
        claim_b = _make_claim("Context windows impose hard limits on recall")
        strand = _make_strand(
            0,
            "Memory Architecture",
            topics=("memory", "persistence"),
            claims=(claim_a, claim_b),
        )
        tensor = _make_rich_tensor(
            preamble="Memory study tensor",
            strands=(strand,),
        )
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201

        # Query by strand topic
        r = client.get("/api/v1/queries/claims-about?topic=memory")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        texts = {c["claim"] for c in data}
        assert "Memory persistence requires structural redundancy" in texts
        assert "Context windows impose hard limits on recall" in texts
        # Each result should carry tensor_id and strand_index
        for item in data:
            assert item["tensor_id"] == str(tensor.id)
            assert item["strand_index"] == 0
            assert "epistemic" in item

    def test_store_then_correct_then_check_status(self, client):
        """Full correction chain: store, correct, check epistemic status."""
        claim = _make_claim("Earth is flat")
        strand = _make_strand(0, "Geography", claims=(claim,), topics=("geography",))
        tensor = _make_rich_tensor(strands=(strand,))
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        correction = CorrectionRecord(
            target_tensor=tensor.id,
            target_strand_index=0,
            target_claim_id=claim.claim_id,
            original_claim="Earth is flat",
            corrected_claim="Earth is an oblate spheroid",
            evidence="Satellite imagery",
        )
        client.post("/api/v1/corrections", json=correction.model_dump(mode="json"))

        r = client.get(f"/api/v1/queries/epistemic-status/{claim.claim_id}")
        data = r.json()
        assert data["correction_count"] == 1
        assert data["current_claim"] == "Earth is an oblate spheroid"
        assert data["original_claim"] == "Earth is flat"

    def test_store_dissent_then_query_disagreements(self, client):
        tensor = _make_rich_tensor()
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        dissent = DissentRecord(
            target_tensor=tensor.id,
            alternative_framework="Competing theory",
            reasoning="The original analysis missed edge cases",
        )
        client.post("/api/v1/dissents", json=dissent.model_dump(mode="json"))

        r = client.get("/api/v1/queries/disagreements")
        data = r.json()
        assert len(data) == 1
        assert data[0]["type"] == "dissent"
        assert data[0]["framework"] == "Competing theory"

    def test_store_negation_then_query_disagreements(self, client):
        t1 = _make_rich_tensor(preamble="A")
        t2 = _make_rich_tensor(preamble="B")
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        negation = NegationRecord(
            tensor_a=t1.id,
            tensor_b=t2.id,
            reasoning="Fundamentally incompatible",
        )
        client.post("/api/v1/negations", json=negation.model_dump(mode="json"))

        r = client.get("/api/v1/queries/disagreements")
        data = r.json()
        negation_entries = [d for d in data if d["type"] == "negation"]
        assert len(negation_entries) == 1
        assert negation_entries[0]["reasoning"] == "Fundamentally incompatible"

    def test_lineage_shared_tags(self, client):
        """Tensors sharing lineage_tags should appear in each other's lineage."""
        t1 = _make_rich_tensor(lineage_tags=("series-A",))
        t2 = _make_rich_tensor(lineage_tags=("series-A", "series-B"))
        t3 = _make_rich_tensor(lineage_tags=("series-B",))
        for t in (t1, t2, t3):
            client.post("/api/v1/tensors", json=t.model_dump(mode="json"))

        r = client.get(f"/api/v1/queries/lineage/{t1.id}")
        data = r.json()
        ids = {d["id"] for d in data}
        assert str(t1.id) in ids  # shares tag with itself
        assert str(t2.id) in ids  # shares "series-A"
        # t3 only has "series-B", doesn't share with t1
        assert str(t3.id) not in ids

    def test_losses_roundtrip(self, client):
        losses = (
            DeclaredLoss(
                what_was_lost="Nuance in emotional register",
                why="Compression for context budget",
                category=LossCategory.CONTEXT_PRESSURE,
            ),
            DeclaredLoss(
                what_was_lost="Alternative interpretation path",
                why="Chose primary reading",
                category=LossCategory.AUTHORIAL_CHOICE,
            ),
        )
        tensor = _make_rich_tensor(declared_losses=losses)
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        r = client.get(f"/api/v1/queries/losses/{tensor.id}")
        data = r.json()
        assert len(data) == 2
        categories = {d["category"] for d in data}
        assert categories == {"context_pressure", "authorial_choice"}


# ---------------------------------------------------------------------------
# 8. Composition edges with real tensor references
# ---------------------------------------------------------------------------


class TestCompositionEdgesWithTensors:
    """Test composition graph queries with stored tensors."""

    def test_composition_graph_with_real_tensors(self, client):
        t1 = _make_rich_tensor(preamble="Source")
        t2 = _make_rich_tensor(preamble="Target")
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        edge = CompositionEdge(
            from_tensor=t1.id,
            to_tensor=t2.id,
            relation_type=RelationType.COMPOSES_WITH,
            ordering=1,
            authored_mapping="Maps strand 0 to strand 1",
        )
        client.post("/api/v1/composition-edges", json=edge.model_dump(mode="json"))

        r = client.get("/api/v1/queries/composition-graph")
        data = r.json()
        assert len(data) == 1
        assert data[0]["from_tensor"] == str(t1.id)
        assert data[0]["to_tensor"] == str(t2.id)
        assert data[0]["relation_type"] == "composes_with"
        assert data[0]["ordering"] == 1

    def test_bridges_only_returns_edges_with_authored_mapping(self, client):
        t1, t2, t3 = uuid4(), uuid4(), uuid4()

        # Edge WITH authored_mapping
        edge_with = CompositionEdge(
            from_tensor=t1,
            to_tensor=t2,
            relation_type=RelationType.COMPOSES_WITH,
            authored_mapping="Explicit bridge mapping",
        )
        # Edge WITHOUT authored_mapping
        edge_without = CompositionEdge(
            from_tensor=t2,
            to_tensor=t3,
            relation_type=RelationType.REFINES,
        )
        client.post("/api/v1/composition-edges", json=edge_with.model_dump(mode="json"))
        client.post("/api/v1/composition-edges", json=edge_without.model_dump(mode="json"))

        r = client.get("/api/v1/queries/bridges")
        data = r.json()
        assert len(data) == 1
        assert data[0]["authored_mapping"] == "Explicit bridge mapping"

    def test_multiple_edges_same_tensors_different_types(self, client):
        """Two edges between the same tensors with different relation types."""
        t1, t2 = uuid4(), uuid4()
        edge_a = CompositionEdge(
            from_tensor=t1, to_tensor=t2,
            relation_type=RelationType.COMPOSES_WITH,
        )
        edge_b = CompositionEdge(
            from_tensor=t1, to_tensor=t2,
            relation_type=RelationType.CORRECTS,
        )
        client.post("/api/v1/composition-edges", json=edge_a.model_dump(mode="json"))
        client.post("/api/v1/composition-edges", json=edge_b.model_dump(mode="json"))

        r = client.get("/api/v1/queries/composition-graph")
        data = r.json()
        assert len(data) == 2
        types = {e["relation_type"] for e in data}
        assert types == {"composes_with", "corrects"}


# ---------------------------------------------------------------------------
# 9. Decoder ring pass-through — UUID integrity
# ---------------------------------------------------------------------------


class TestDecoderRing:
    """Verify the decoder ring doesn't corrupt UUIDs during pass-through."""

    def test_decoder_encode_preserves_uuid(self):
        decoder = DecoderRing()
        original = uuid4()
        encoded = decoder.encode(original)
        assert encoded == original
        assert type(encoded) is UUID

    def test_decoder_decode_preserves_uuid(self):
        decoder = DecoderRing()
        original = uuid4()
        decoded = decoder.decode(original)
        assert decoded == original
        assert type(decoded) is UUID

    def test_decoder_roundtrip(self):
        decoder = DecoderRing()
        original = uuid4()
        assert decoder.decode(decoder.encode(original)) == original
        assert decoder.encode(decoder.decode(original)) == original

    def test_decoder_preserves_specific_uuid_format(self):
        """A known UUID should survive encode/decode unchanged."""
        decoder = DecoderRing()
        known = UUID("12345678-1234-5678-1234-567812345678")
        assert decoder.encode(known) == known
        assert decoder.decode(known) == known
        assert str(decoder.encode(known)) == "12345678-1234-5678-1234-567812345678"

    def test_stored_uuid_matches_returned_uuid(self, client):
        """Store a tensor, retrieve it, verify UUIDs survived the HTTP layer."""
        tensor = _make_rich_tensor()
        original_id = tensor.id
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        returned_id = UUID(r.json()["id"])
        assert returned_id == original_id

        r = client.get(f"/api/v1/tensors/{original_id}")
        retrieved_id = UUID(r.json()["id"])
        assert retrieved_id == original_id

    def test_claim_uuids_survive_roundtrip(self, client):
        """Key claim UUIDs should survive store -> read -> query."""
        claim = _make_claim("Test claim for UUID integrity")
        strand = _make_strand(0, "UUID Test", claims=(claim,), topics=("uuid-test",))
        tensor = _make_rich_tensor(strands=(strand,))
        client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))

        # Read the tensor back and check claim_id
        r = client.get(f"/api/v1/tensors/{tensor.id}")
        stored_claim_id = UUID(r.json()["strands"][0]["key_claims"][0]["claim_id"])
        assert stored_claim_id == claim.claim_id

    def test_entity_uuid_survives_roundtrip(self, client):
        """entity_uuid field should survive store -> read."""
        entity_uuid = uuid4()
        entity = EntityResolution(
            entity_uuid=entity_uuid,
            identity_type="test",
            identity_data={"role": "verifier"},
        )
        client.post("/api/v1/entities", json=entity.model_dump(mode="json"))

        r = client.get(f"/api/v1/entities/{entity.id}")
        assert UUID(r.json()["entity_uuid"]) == entity_uuid

    def test_composition_edge_uuids_survive_roundtrip(self, client):
        """from_tensor and to_tensor UUIDs should survive store -> query."""
        from_id, to_id = uuid4(), uuid4()
        edge = CompositionEdge(
            from_tensor=from_id,
            to_tensor=to_id,
            relation_type=RelationType.COMPOSES_WITH,
        )
        client.post("/api/v1/composition-edges", json=edge.model_dump(mode="json"))

        r = client.get("/api/v1/queries/composition-graph")
        data = r.json()
        assert UUID(data[0]["from_tensor"]) == from_id
        assert UUID(data[0]["to_tensor"]) == to_id


# ---------------------------------------------------------------------------
# 10. Meta endpoint tests
# ---------------------------------------------------------------------------


class TestMetaEndpoints:
    """Verify health, version, and counts."""

    def test_health_returns_pukara(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body == {"status": "ok", "service": "pukara"}

    def test_version_returns_gateway_and_interface(self, client):
        r = client.get("/api/v1/version")
        assert r.status_code == 200
        body = r.json()
        assert body["gateway"] == "0.1.0"
        assert body["interface"] == "v1"

    def test_counts_all_zeros_on_fresh_backend(self, client):
        r = client.get("/api/v1/counts")
        assert r.status_code == 200
        data = r.json()
        expected_keys = {
            "tensors", "edges", "corrections", "dissents",
            "negations", "bootstraps", "evolutions", "entities",
        }
        assert set(data.keys()) == expected_keys
        for key in expected_keys:
            assert data[key] == 0

    def test_counts_increment_after_stores(self, client):
        # Store one of each type
        client.post("/api/v1/tensors", json=TensorRecord().model_dump(mode="json"))
        client.post(
            "/api/v1/composition-edges",
            json=CompositionEdge(
                from_tensor=uuid4(), to_tensor=uuid4(),
                relation_type=RelationType.COMPOSES_WITH,
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/corrections",
            json=CorrectionRecord(
                target_tensor=uuid4(), original_claim="A", corrected_claim="B",
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/dissents",
            json=DissentRecord(
                target_tensor=uuid4(), alternative_framework="X", reasoning="Y",
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/negations",
            json=NegationRecord(
                tensor_a=uuid4(), tensor_b=uuid4(), reasoning="Z",
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/bootstraps",
            json=BootstrapRecord(
                instance_id="t", context_budget=1.0,
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/evolutions",
            json=SchemaEvolutionRecord(
                from_version="v0", to_version="v1",
            ).model_dump(mode="json"),
        )
        client.post(
            "/api/v1/entities",
            json=EntityResolution(
                entity_uuid=uuid4(), identity_type="t", identity_data={},
            ).model_dump(mode="json"),
        )

        r = client.get("/api/v1/counts")
        data = r.json()
        assert data["tensors"] == 1
        assert data["edges"] == 1
        assert data["corrections"] == 1
        assert data["dissents"] == 1
        assert data["negations"] == 1
        assert data["bootstraps"] == 1
        assert data["evolutions"] == 1
        assert data["entities"] == 1


# ---------------------------------------------------------------------------
# 11. Cross-model query requires multiple model families
# ---------------------------------------------------------------------------


class TestCrossModelQuery:
    """cross-model query should return empty when only one family, all when multiple."""

    def test_single_family_returns_empty(self, client):
        t1 = _make_rich_tensor(model_family="claude-opus")
        t2 = _make_rich_tensor(model_family="claude-opus")
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        r = client.get("/api/v1/queries/cross-model")
        assert r.status_code == 200
        assert r.json() == []

    def test_multiple_families_returns_all(self, client):
        t1 = _make_rich_tensor(model_family="claude-opus")
        t2 = _make_rich_tensor(model_family="gpt-4o")
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        r = client.get("/api/v1/queries/cross-model")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

    def test_empty_family_not_counted(self, client):
        """Tensors with empty author_model_family shouldn't count as a family."""
        t1 = _make_rich_tensor(model_family="claude-opus")
        t2 = TensorRecord()  # empty model family
        client.post("/api/v1/tensors", json=t1.model_dump(mode="json"))
        client.post("/api/v1/tensors", json=t2.model_dump(mode="json"))

        r = client.get("/api/v1/queries/cross-model")
        assert r.status_code == 200
        assert r.json() == []  # only one non-empty family


# ---------------------------------------------------------------------------
# 12. Isolation: each test gets a fresh backend
# ---------------------------------------------------------------------------


class TestIsolation:
    """Verify that the fixture provides a clean backend per test."""

    def test_first_store(self, client):
        tensor = TensorRecord(preamble="isolation test 1")
        r = client.post("/api/v1/tensors", json=tensor.model_dump(mode="json"))
        assert r.status_code == 201
        assert client.get("/api/v1/counts").json()["tensors"] == 1

    def test_second_store_sees_empty(self, client):
        """If isolation works, this test starts from zero."""
        assert client.get("/api/v1/counts").json()["tensors"] == 0
