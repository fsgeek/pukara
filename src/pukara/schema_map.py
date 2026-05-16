"""SchemaMap: structural obfuscation for storage labels.

Maps semantic labels (collection names, field names) to opaque
UUID-derived identifiers at the storage boundary. Application code
uses semantic names; backends store opaque names.

Production: UUID5-derived names. The namespace UUID is the
per-deployment secret (from PUKARA_STORAGE_KEY env var).
Same namespace + same name = same opaque name (deterministic).
Different namespace = different mapping (per-deployment isolation).

Development: identity mapping (the transparent shim). Removing
the shim for production subtracts capability --- safer than adding.

Content encryption is NOT included --- that requires designing around
the indexing constraint (order-preserving encryption, application-level
encrypt/decrypt, or encrypted search). Structural obfuscation makes
the database not self-documenting. A determined reader can still
interpret content values. That's a declared loss.

This module lives in Pukara because the adversary is the database
provider, not a compromised device. The fortress is the trust boundary.
Obfuscation happens here, not in the backend library.

Uses stdlib only (uuid). No new dependencies.
"""

from __future__ import annotations

import os
from typing import Literal
from uuid import UUID, uuid5


# ArangoDB internal keys --- structural, not semantic. Never mapped.
_ARANGO_META = frozenset({"_key", "_id", "_rev", "_from", "_to"})


def _collect_model_field_names() -> set[str]:
    """Walk all known Pydantic models and collect their field names.

    Populates the bidirectional cache so that deobfuscation works
    across process restarts. Every field name from every model
    in both Apacheta and Activity gets registered.
    """
    from yanantin.activity.models import AnchorCursor, FactRecord, MemoryAnchor

    from tiksi.composition import (
        BootstrapRecord,
        CompositionEdge,
        CorrectionRecord,
        DissentRecord,
        NegationRecord,
        SchemaEvolutionRecord,
    )
    from tiksi.entities import EntityResolution
    from tiksi.epistemics import DeclaredLoss, EpistemicMetadata
    from tiksi.provenance import ProvenanceEnvelope, SourceIdentifier
    from tiksi.tensor import KeyClaim, StrandRecord, TensorRecord

    names: set[str] = set()
    for model in (
        TensorRecord,
        StrandRecord,
        KeyClaim,
        CompositionEdge,
        CorrectionRecord,
        DissentRecord,
        NegationRecord,
        BootstrapRecord,
        SchemaEvolutionRecord,
        EntityResolution,
        ProvenanceEnvelope,
        SourceIdentifier,
        EpistemicMetadata,
        DeclaredLoss,
        FactRecord,
        MemoryAnchor,
        AnchorCursor,
    ):
        names.update(model.model_fields)
    return names


class SchemaMap:
    """Maps semantic labels to opaque storage identifiers.

    Production: UUID5-derived names for collections and fields.
    Development: identity mapping (the removable shim).

    Satisfies yanantin's StorageObfuscator protocol (duck-typed).
    """

    def __init__(
        self,
        namespace: UUID,
        mode: Literal["opaque", "transparent"] = "opaque",
    ) -> None:
        self._namespace = namespace
        self._mode = mode
        self._field_cache: dict[str, str] = {}  # semantic -> opaque
        self._reverse_cache: dict[str, str] = {}  # opaque -> semantic

        if mode == "opaque":
            for name in _collect_model_field_names():
                self._register_field(name)

    @classmethod
    def transparent(cls) -> SchemaMap:
        """Dev/test shim --- identity mapping."""
        return cls(UUID(int=0), mode="transparent")

    @classmethod
    def from_env(cls) -> SchemaMap:
        """Production --- key from PUKARA_STORAGE_KEY.

        Fail-stop if the env var is missing or not a valid UUID.
        """
        raw = os.environ.get("PUKARA_STORAGE_KEY")
        if raw is None:
            raise RuntimeError(
                "PUKARA_STORAGE_KEY not set. "
                "Opaque storage requires a UUID namespace key. "
                "Set the env var or use SchemaMap.transparent() for development."
            )
        try:
            namespace = UUID(raw)
        except ValueError as e:
            raise RuntimeError(
                f"PUKARA_STORAGE_KEY is not a valid UUID: {raw!r}"
            ) from e
        return cls(namespace, mode="opaque")

    def collection_name(self, semantic: str) -> str:
        """Map a semantic collection name to an opaque identifier.

        Returns ``c_`` + uuid5 hex (34 chars total). ArangoDB requires
        collection names to start with a letter.
        """
        if self._mode == "transparent":
            return semantic
        derived = uuid5(self._namespace, semantic)
        return f"c_{derived.hex}"

    def field_name(self, semantic: str) -> str:
        """Map a semantic field name to an opaque identifier.

        ArangoDB internal keys (_key, _id, _rev) pass through unchanged.
        Returns ``f_`` + uuid5 hex (34 chars total).

        Unknown field names are computed and cached on the fly.
        """
        if self._mode == "transparent":
            return semantic
        if semantic in _ARANGO_META:
            return semantic
        if semantic not in self._field_cache:
            self._register_field(semantic)
        return self._field_cache[semantic]

    def reverse_field(self, opaque: str) -> str:
        """Reverse lookup from opaque to semantic field name.

        Raises KeyError if the opaque name is not in the cache.
        ArangoDB internal keys pass through unchanged.
        """
        if self._mode == "transparent":
            return opaque
        if opaque in _ARANGO_META:
            return opaque
        if opaque not in self._reverse_cache:
            raise KeyError(
                f"Unknown opaque field name: {opaque!r}. "
                "Field was not pre-registered from any known model."
            )
        return self._reverse_cache[opaque]

    def obfuscate_document(self, doc: dict) -> dict:
        """Recursively rename known dict keys using the field cache.

        Values are unchanged. Nested dicts have their keys mapped.
        Lists containing dicts have those dicts' keys mapped.
        ArangoDB metadata keys (_key, _id, _rev) are preserved.
        Keys not in the field cache are left unchanged --- this is
        intentional for arbitrary-dict fields like identity_data.
        """
        if self._mode == "transparent":
            return doc
        return self._obfuscate_recursive(doc)

    def deobfuscate_document(self, doc: dict) -> dict:
        """Recursively restore known dict keys using the reverse cache.

        Inverse of obfuscate_document(). Keys not in the reverse
        cache are left unchanged.
        """
        if self._mode == "transparent":
            return doc
        return self._deobfuscate_recursive(doc)

    @property
    def is_transparent(self) -> bool:
        """True if in dev/test shim mode."""
        return self._mode == "transparent"

    # -- Internal ----------------------------------------------------------

    def _register_field(self, semantic: str) -> None:
        """Compute and cache the opaque name for a semantic field."""
        derived = uuid5(self._namespace, semantic)
        opaque = f"f_{derived.hex}"
        self._field_cache[semantic] = opaque
        self._reverse_cache[opaque] = semantic

    def _obfuscate_recursive(self, doc: dict) -> dict:
        """Map known keys to opaque form. Unknown keys pass through."""
        result = {}
        for k, v in doc.items():
            if k in _ARANGO_META:
                mapped_key = k
            elif k in self._field_cache:
                mapped_key = self._field_cache[k]
            else:
                mapped_key = k

            if isinstance(v, dict):
                result[mapped_key] = self._obfuscate_recursive(v)
            elif isinstance(v, list):
                result[mapped_key] = [
                    self._obfuscate_recursive(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[mapped_key] = v
        return result

    def _deobfuscate_recursive(self, doc: dict) -> dict:
        """Map opaque keys back to semantic form. Unknown keys pass through."""
        result = {}
        for k, v in doc.items():
            if k in _ARANGO_META:
                semantic_key = k
            elif k in self._reverse_cache:
                semantic_key = self._reverse_cache[k]
            else:
                semantic_key = k

            if isinstance(v, dict):
                result[semantic_key] = self._deobfuscate_recursive(v)
            elif isinstance(v, list):
                result[semantic_key] = [
                    self._deobfuscate_recursive(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[semantic_key] = v
        return result
