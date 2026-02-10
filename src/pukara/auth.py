"""API key authentication for Pukara gateway.

Empty key = development mode (no auth required).
Non-empty key = must match X-API-Key header.
"""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def make_api_key_checker(expected_key: str):
    """Return a FastAPI dependency that checks the API key.

    If expected_key is empty, all requests are allowed (development mode).
    """

    async def check_api_key(
        api_key: str | None = Security(_api_key_header),
    ) -> str | None:
        if not expected_key:
            return None
        if api_key != expected_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
            )
        return api_key

    return check_api_key
