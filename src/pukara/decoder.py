"""The decoder ring — UUID obfuscation between agents and storage.

V1: pass-through. The architecture is in place; actual obfuscation
comes when the provider threat model demands it.

The decoder sits between HTTP endpoints and the backend. External
UUIDs (what agents see) and internal UUIDs (what's in ArangoDB) are
the same in v1 but will diverge when we add obfuscation.

Future:
- Deterministic mapping (HMAC-based UUID derivation with secret key)
- Model-aware UUID traversal (walk Pydantic models, transform all UUID fields)
- Bidirectional lookup table for reverse mapping
"""

from __future__ import annotations

from uuid import UUID


class DecoderRing:
    """UUID obfuscation layer. V1 = pass-through."""

    def encode(self, external_uuid: UUID) -> UUID:
        """Map agent-visible UUID to storage UUID."""
        return external_uuid

    def decode(self, internal_uuid: UUID) -> UUID:
        """Map storage UUID to agent-visible UUID."""
        return internal_uuid
