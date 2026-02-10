"""FastAPI dependencies for Pukara routes."""

from __future__ import annotations

from fastapi import Request

from yanantin.apacheta.interface.abstract import ApachetaInterface

from pukara.decoder import DecoderRing


def get_backend(request: Request) -> ApachetaInterface:
    """Get the storage backend from app state."""
    return request.app.state.backend


def get_decoder(request: Request) -> DecoderRing:
    """Get the decoder ring from app state."""
    return request.app.state.decoder
