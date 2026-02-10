"""Meta endpoints — health, version, record counts."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from yanantin.apacheta.interface.abstract import ApachetaInterface

from pukara.deps import get_backend

router = APIRouter(prefix="/api/v1", tags=["meta"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "pukara"}


@router.get("/version")
def version(backend: ApachetaInterface = Depends(get_backend)):
    return {
        "gateway": "0.1.0",
        "interface": backend.get_interface_version(),
    }


@router.get("/counts")
def counts(backend: ApachetaInterface = Depends(get_backend)):
    return backend.count_records()
