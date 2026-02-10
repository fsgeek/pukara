"""Store endpoints — write operations that produce new records.

All produce new records. None modify existing ones.
Immutability enforced by the backend: duplicate UUID → 409.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from yanantin.apacheta.interface.abstract import ApachetaInterface
from yanantin.apacheta.models.composition import (
    BootstrapRecord,
    CompositionEdge,
    CorrectionRecord,
    DissentRecord,
    NegationRecord,
    SchemaEvolutionRecord,
)
from yanantin.apacheta.models.entities import EntityResolution
from yanantin.apacheta.models.tensor import TensorRecord

from pukara.deps import get_backend

router = APIRouter(prefix="/api/v1", tags=["store"])


@router.post("/tensors", status_code=201)
def store_tensor(
    tensor: TensorRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_tensor(tensor)
    return {"id": str(tensor.id)}


@router.post("/composition-edges", status_code=201)
def store_composition_edge(
    edge: CompositionEdge,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_composition_edge(edge)
    return {"id": str(edge.id)}


@router.post("/corrections", status_code=201)
def store_correction(
    correction: CorrectionRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_correction(correction)
    return {"id": str(correction.id)}


@router.post("/dissents", status_code=201)
def store_dissent(
    dissent: DissentRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_dissent(dissent)
    return {"id": str(dissent.id)}


@router.post("/negations", status_code=201)
def store_negation(
    negation: NegationRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_negation(negation)
    return {"id": str(negation.id)}


@router.post("/bootstraps", status_code=201)
def store_bootstrap(
    bootstrap: BootstrapRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_bootstrap(bootstrap)
    return {"id": str(bootstrap.id)}


@router.post("/evolutions", status_code=201)
def store_evolution(
    evolution: SchemaEvolutionRecord,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_evolution(evolution)
    return {"id": str(evolution.id)}


@router.post("/entities", status_code=201)
def store_entity(
    entity: EntityResolution,
    backend: ApachetaInterface = Depends(get_backend),
):
    backend.store_entity(entity)
    return {"id": str(entity.id)}
