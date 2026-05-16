"""FastAPI dependencies for Pukara routes."""

from __future__ import annotations

from fastapi import Request

from yanantin.apacheta.interface.abstract import ApachetaInterface


def get_backend(request: Request) -> ApachetaInterface:
    """Get the storage backend from app state."""
    return request.app.state.backend
