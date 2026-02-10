"""Query endpoints — 20 query operations across 7 categories."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from yanantin.apacheta.interface.abstract import ApachetaInterface

from pukara.deps import get_backend

router = APIRouter(prefix="/api/v1/queries", tags=["query"])


# ── Bootstrap queries ──────────────────────────────────────────


@router.get("/tensors-for-budget")
def tensors_for_budget(
    budget: float = Query(...),
    backend: ApachetaInterface = Depends(get_backend),
):
    tensors = backend.query_tensors_for_budget(budget)
    return [t.model_dump(mode="json") for t in tensors]


@router.get("/operational-principles")
def operational_principles(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_operational_principles()


@router.get("/project-state")
def project_state(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_project_state()


# ── Epistemic queries ─────────────────────────────────────────


@router.get("/claims-about")
def claims_about(
    topic: str = Query(...),
    backend: ApachetaInterface = Depends(get_backend),
):
    return backend.query_claims_about(topic)


@router.get("/correction-chain/{claim_id}")
def correction_chain(
    claim_id: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    corrections = backend.query_correction_chain(claim_id)
    return [c.model_dump(mode="json") for c in corrections]


@router.get("/epistemic-status/{claim_id}")
def epistemic_status(
    claim_id: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    return backend.query_epistemic_status(claim_id)


@router.get("/disagreements")
def disagreements(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_disagreements()


# ── Lineage queries ───────────────────────────────────────────


@router.get("/composition-graph")
def composition_graph(backend: ApachetaInterface = Depends(get_backend)):
    edges = backend.query_composition_graph()
    return [e.model_dump(mode="json") for e in edges]


@router.get("/lineage/{tensor_id}")
def lineage(
    tensor_id: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    tensors = backend.query_lineage(tensor_id)
    return [t.model_dump(mode="json") for t in tensors]


@router.get("/bridges")
def bridges(backend: ApachetaInterface = Depends(get_backend)):
    edges = backend.query_bridges()
    return [e.model_dump(mode="json") for e in edges]


# ── Evolution queries ─────────────────────────────────────────


@router.get("/error-classes")
def error_classes(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_error_classes()


@router.get("/open-questions")
def open_questions(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_open_questions()


@router.get("/unreliable-signals")
def unreliable_signals(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_unreliable_signals()


@router.get("/anti-patterns")
def anti_patterns(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_anti_patterns()


# ── Provenance queries ────────────────────────────────────────


@router.get("/authorship/{tensor_id}")
def authorship(
    tensor_id: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    return backend.query_authorship(tensor_id)


@router.get("/cross-model")
def cross_model(backend: ApachetaInterface = Depends(get_backend)):
    tensors = backend.query_cross_model()
    return [t.model_dump(mode="json") for t in tensors]


@router.get("/reading-order")
def reading_order(
    tag: str = Query(...),
    backend: ApachetaInterface = Depends(get_backend),
):
    tensors = backend.query_reading_order(tag)
    return [t.model_dump(mode="json") for t in tensors]


# ── Defensive queries ─────────────────────────────────────────


@router.get("/unlearn")
def unlearn(
    topic: str = Query(...),
    backend: ApachetaInterface = Depends(get_backend),
):
    return backend.query_unlearn(topic)


# ── Loss queries ──────────────────────────────────────────────


@router.get("/losses/{tensor_id}")
def losses(
    tensor_id: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    return backend.query_losses(tensor_id)


@router.get("/loss-patterns")
def loss_patterns(backend: ApachetaInterface = Depends(get_backend)):
    return backend.query_loss_patterns()


# ── Entity queries ────────────────────────────────────────────


@router.get("/entities-by-uuid/{entity_uuid}")
def entities_by_uuid(
    entity_uuid: UUID,
    backend: ApachetaInterface = Depends(get_backend),
):
    entities = backend.query_entities_by_uuid(entity_uuid)
    return [e.model_dump(mode="json") for e in entities]
