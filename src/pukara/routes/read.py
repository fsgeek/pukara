"""Read endpoints — retrieve individual records."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from yanantin.apacheta.interface.abstract import ApachetaInterface

from pukara.deps import get_backend

router = APIRouter(prefix="/api/v1", tags=["read"])


@router.get("/tensors")
def list_tensors(backend: ApachetaInterface = Depends(get_backend)):
    tensors = backend.list_tensors()
    return [t.model_dump(mode="json") for t in tensors]


@router.get("/tensors/{tensor_id}")
def get_tensor(tensor_id: UUID, backend: ApachetaInterface = Depends(get_backend)):
    tensor = backend.get_tensor(tensor_id)
    return tensor.model_dump(mode="json")


@router.get("/tensors/{tensor_id}/strands/{strand_index}")
def get_strand(
    tensor_id: UUID,
    strand_index: int,
    backend: ApachetaInterface = Depends(get_backend),
):
    tensor = backend.get_strand(tensor_id, strand_index)
    return tensor.model_dump(mode="json")


@router.get("/entities/{entity_id}")
def get_entity(entity_id: UUID, backend: ApachetaInterface = Depends(get_backend)):
    entity = backend.get_entity(entity_id)
    return entity.model_dump(mode="json")
