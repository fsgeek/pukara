"""Llika graph endpoints — link / walk / neighbors / get behind the fortress.

Routes Llika's graph surface through Pukara so LlikaService stops holding
its own privileged ArangoDB handle (yanantin#10 slice 1). The verbs call
the GraphBackend capability — a narrow protocol the ArangoDBBackend
implements, deliberately kept OFF the public ApachetaInterface catalog
(the find spec forbids polluting it). See
docs/superpowers/specs/2026-06-06-llika-behind-pukara-design.md and the
work-order contract under docs/superpowers/contracts/.

`find` is intentionally absent — a callable predicate cannot cross a wire.

The dependency is typed against a local `Backend` alias (see below) rather
than ApachetaInterface, because the graph verbs deliberately do not live
on ApachetaInterface. The routes are correct against the contract's
signatures today; they execute once the yanantin-side backend implements
the verbs.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from yanantin.apacheta.models import ProvenanceEnvelope
from yanantin.apacheta.models.composition import RelationType

from pukara.deps import get_backend

router = APIRouter(prefix="/api/v1/llika", tags=["llika"])

# The live backend (ArangoDBBackend) satisfies the GraphBackend capability
# — link/walk/neighbors/get_record — which is deliberately NOT on the
# public ApachetaInterface catalog (see module docstring + the contract).
# That protocol is built yanantin-side; until it is importable here, the
# dependency is typed `Backend = Any` so the routes type-check against the
# contract's verb signatures without falsely claiming they live on
# ApachetaInterface. This alias tightens to the real GraphBackend protocol
# once yanantin lands it. The looseness is named, not hidden.
Backend = Any


# ── Request bodies (the wire contract) ────────────────────────────


class LinkRequest(BaseModel):
    """Create one immutable edge from_ref -> to_ref.

    provenance is agent-supplied and passed through verbatim — TRANSPORT
    trust, not truth trust. Pukara is not the author and does not verify
    authorship; the stored edge carries authorship_verified=False (the
    default on ProvenanceEnvelope) so a future reader sees the claim is
    unverified IN THE DATA, not merely in a test. See yanantin#13.
    """

    from_ref: str            # "collection/<uuid>" vertex ref
    to_ref: str              # "collection/<uuid>" vertex ref
    relation_type: RelationType
    provenance: ProvenanceEnvelope


class WalkRequest(BaseModel):
    """Traverse from start_id by structure: direction + depth + filter."""

    start_id: str            # "collection/<uuid>"
    direction: str           # "forward" | "backward" | "both"
    depth: int
    relation_types: list[str] | None = None   # RelationType VALUES
    max_results: int = 50


class NeighborsRequest(BaseModel):
    """Depth-1 adjacency. walk(..., depth=1)."""

    start_id: str
    direction: str
    relation_types: list[str] | None = None


# ── Routes ────────────────────────────────────────────────────────


@router.post("/link", status_code=201)
def link(
    req: LinkRequest,
    backend: Backend = Depends(get_backend),
):
    result = backend.link(
        req.from_ref,
        req.to_ref,
        req.relation_type,
        req.provenance,
    )
    return _edge_result_json(result)


@router.post("/walk")
def walk(
    req: WalkRequest,
    backend: Backend = Depends(get_backend),
):
    results = backend.walk(
        req.start_id,
        req.direction,
        req.depth,
        req.relation_types,
        req.max_results,
    )
    return [_path_result_json(r) for r in results]


@router.post("/neighbors")
def neighbors(
    req: NeighborsRequest,
    backend: Backend = Depends(get_backend),
):
    results = backend.neighbors(
        req.start_id,
        req.direction,
        req.relation_types,
    )
    return [_path_result_json(r) for r in results]


@router.get("/get/{record_id}")
def get(record_id: UUID, backend: Backend = Depends(get_backend)):
    # Rides the existing interface method — no new result type.
    record = backend.get_record(record_id)
    return record.model_dump(mode="json")


# ── Serialization of the frozen-dataclass result types ────────────
#
# EdgeResult / PathResult / PathStep (yanantin/llika/models.py) are frozen
# dataclasses, not pydantic models — serialize their fields explicitly so
# the wire shape is stable and obvious rather than relying on dataclass
# internals.


def _edge_result_json(r) -> dict:
    return {
        "edge_id": r.edge_id,
        "from_id": r.from_id,
        "to_id": r.to_id,
        "relation_type": r.relation_type,
        "created_at": r.created_at,
    }


def _path_result_json(r) -> dict:
    return {
        "start_id": r.start_id,
        "steps": [
            {
                "record_id": s.record_id,
                "relation_type": s.relation_type,
                "field_names": list(s.field_names),
            }
            for s in r.steps
        ],
    }
