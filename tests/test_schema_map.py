"""Unit tests for SchemaMap --- structural obfuscation.

SchemaMap lives in Pukara because the adversary is the database provider,
not a compromised device. The fortress is the trust boundary.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from pukara.schema_map import SchemaMap, _ARANGO_META


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def namespace() -> UUID:
    """A fixed namespace for deterministic tests."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def opaque_map(namespace: UUID) -> SchemaMap:
    return SchemaMap(namespace, mode="opaque")


@pytest.fixture
def transparent_map() -> SchemaMap:
    return SchemaMap.transparent()


# ── Opaque mode ──────────────────────────────────────────────────────

def test_opaque_names_are_not_human_readable(opaque_map: SchemaMap):
    """Collection and field names must not match their semantic input."""
    assert opaque_map.collection_name("tensors") != "tensors"
    assert opaque_map.field_name("author_model_family") != "author_model_family"
    assert opaque_map.field_name("provider_id") != "provider_id"

    # Must start with the correct prefix
    assert opaque_map.collection_name("tensors").startswith("c_")
    assert opaque_map.field_name("author_model_family").startswith("f_")


def test_opaque_collection_name_length(opaque_map: SchemaMap):
    """Opaque collection names are c_ + 32 hex chars = 34 chars."""
    name = opaque_map.collection_name("tensors")
    assert len(name) == 34
    # Hex chars only after prefix
    assert all(c in "0123456789abcdef" for c in name[2:])


def test_opaque_field_name_length(opaque_map: SchemaMap):
    """Opaque field names are f_ + 32 hex chars = 34 chars."""
    name = opaque_map.field_name("strand_index")
    assert len(name) == 34
    assert all(c in "0123456789abcdef" for c in name[2:])


# ── Transparent mode ─────────────────────────────────────────────────

def test_transparent_mode_is_identity(transparent_map: SchemaMap):
    """All methods return their input unchanged in transparent mode."""
    assert transparent_map.collection_name("tensors") == "tensors"
    assert transparent_map.field_name("author_model_family") == "author_model_family"
    assert transparent_map.reverse_field("author_model_family") == "author_model_family"
    assert transparent_map.is_transparent is True


def test_transparent_obfuscate_is_identity(transparent_map: SchemaMap):
    """obfuscate_document returns the same dict in transparent mode."""
    doc = {"id": "abc", "provenance": {"timestamp": "2025-01-01"}}
    assert transparent_map.obfuscate_document(doc) is doc


def test_transparent_deobfuscate_is_identity(transparent_map: SchemaMap):
    """deobfuscate_document returns the same dict in transparent mode."""
    doc = {"id": "abc", "provenance": {"timestamp": "2025-01-01"}}
    assert transparent_map.deobfuscate_document(doc) is doc


# ── Roundtrip ────────────────────────────────────────────────────────

def test_roundtrip_obfuscate_deobfuscate(opaque_map: SchemaMap):
    """doc -> obfuscate -> deobfuscate -> original doc."""
    doc = {
        "_key": "some-uuid",
        "provenance": {
            "source": {"identifier": "src-uuid", "version": "v1"},
            "timestamp": "2025-01-01T00:00:00Z",
            "author_model_family": "claude",
        },
        "preamble": "Hello",
        "strands": [
            {
                "strand_index": 0,
                "title": "First strand",
                "content": "Some content",
                "topics": ["topic1"],
                "key_claims": [
                    {"claim_id": "claim-uuid", "text": "A claim", "evidence_refs": []}
                ],
            }
        ],
        "lineage_tags": ["tag1", "tag2"],
        "declared_losses": [],
    }

    obfuscated = opaque_map.obfuscate_document(doc)
    restored = opaque_map.deobfuscate_document(obfuscated)
    assert restored == doc


def test_roundtrip_field_name(opaque_map: SchemaMap):
    """field_name -> reverse_field recovers the semantic name."""
    semantic = "author_model_family"
    opaque = opaque_map.field_name(semantic)
    assert opaque_map.reverse_field(opaque) == semantic


# ── Nested documents ─────────────────────────────────────────────────

def test_nested_documents_handled(opaque_map: SchemaMap):
    """Strands contain nested dicts with topics, claims, epistemic."""
    doc = {
        "strands": [
            {
                "strand_index": 0,
                "title": "Test",
                "epistemic": {
                    "truth": 0.8,
                    "indeterminacy": 0.1,
                    "falsity": 0.1,
                },
            }
        ],
    }

    obfuscated = opaque_map.obfuscate_document(doc)

    # Top-level key should be mapped
    assert "strands" not in obfuscated
    # The mapped key's value is a list with one dict
    values = list(obfuscated.values())
    assert len(values) == 1
    strand_list = values[0]
    assert isinstance(strand_list, list)
    assert len(strand_list) == 1

    strand = strand_list[0]
    # Nested keys should also be mapped
    assert "strand_index" not in strand
    assert "title" not in strand
    assert "epistemic" not in strand

    # Values should be preserved
    restored = opaque_map.deobfuscate_document(obfuscated)
    assert restored == doc


# ── Per-deployment isolation ─────────────────────────────────────────

def test_different_keys_produce_different_mappings():
    """Different namespace UUIDs produce different opaque names."""
    ns1 = UUID("11111111-1111-1111-1111-111111111111")
    ns2 = UUID("22222222-2222-2222-2222-222222222222")

    map1 = SchemaMap(ns1, mode="opaque")
    map2 = SchemaMap(ns2, mode="opaque")

    assert map1.collection_name("tensors") != map2.collection_name("tensors")
    assert map1.field_name("author_model_family") != map2.field_name("author_model_family")


# ── Fail-stop ────────────────────────────────────────────────────────

def test_missing_key_in_opaque_mode_fails(monkeypatch: pytest.MonkeyPatch):
    """from_env() fails if PUKARA_STORAGE_KEY is not set."""
    monkeypatch.delenv("PUKARA_STORAGE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PUKARA_STORAGE_KEY not set"):
        SchemaMap.from_env()


def test_invalid_key_in_opaque_mode_fails(monkeypatch: pytest.MonkeyPatch):
    """from_env() fails if PUKARA_STORAGE_KEY is not a valid UUID."""
    monkeypatch.setenv("PUKARA_STORAGE_KEY", "not-a-uuid")
    with pytest.raises(RuntimeError, match="not a valid UUID"):
        SchemaMap.from_env()


def test_from_env_with_valid_key(monkeypatch: pytest.MonkeyPatch):
    """from_env() succeeds with a valid UUID."""
    key = str(uuid4())
    monkeypatch.setenv("PUKARA_STORAGE_KEY", key)
    schema_map = SchemaMap.from_env()
    assert not schema_map.is_transparent


def test_reverse_field_unknown_raises(opaque_map: SchemaMap):
    """reverse_field raises KeyError for unregistered opaque names."""
    with pytest.raises(KeyError, match="Unknown opaque field name"):
        opaque_map.reverse_field("f_0000000000000000000000000000dead")


# ── ArangoDB metadata preserved ──────────────────────────────────────

def test_arango_metadata_keys_preserved(opaque_map: SchemaMap):
    """_key, _id, _rev are never mapped."""
    for meta_key in ("_key", "_id", "_rev"):
        assert opaque_map.field_name(meta_key) == meta_key

    doc = {
        "_key": "some-uuid",
        "_id": "collection/some-uuid",
        "_rev": "12345",
        "title": "A tensor",
    }
    obfuscated = opaque_map.obfuscate_document(doc)
    assert obfuscated["_key"] == "some-uuid"
    assert obfuscated["_id"] == "collection/some-uuid"
    assert obfuscated["_rev"] == "12345"
    # But semantic keys ARE mapped
    assert "title" not in obfuscated


# ── Determinism ──────────────────────────────────────────────────────

def test_deterministic(namespace: UUID):
    """Same input + same key = same output every time."""
    map1 = SchemaMap(namespace, mode="opaque")
    map2 = SchemaMap(namespace, mode="opaque")

    assert map1.collection_name("tensors") == map2.collection_name("tensors")
    assert map1.field_name("author_model_family") == map2.field_name("author_model_family")

    doc = {"provenance": {"timestamp": "2025-01-01"}, "preamble": "Hello"}
    assert map1.obfuscate_document(doc) == map2.obfuscate_document(doc)


# ── Arbitrary dict fields ────────────────────────────────────────────

def test_unknown_keys_pass_through_in_obfuscate(opaque_map: SchemaMap):
    """Keys not in any model schema are left unchanged.

    This is important for fields like identity_data and FactRecord.data
    which contain arbitrary user content.
    """
    doc = {
        "title": "Known field",
        "custom_user_key": "arbitrary value",
        "another_unknown": {"nested_unknown": 42},
    }
    obfuscated = opaque_map.obfuscate_document(doc)

    # Known key is mapped
    assert "title" not in obfuscated

    # Unknown keys pass through unchanged
    assert "custom_user_key" in obfuscated
    assert obfuscated["custom_user_key"] == "arbitrary value"
    assert "another_unknown" in obfuscated
    assert obfuscated["another_unknown"] == {"nested_unknown": 42}


def test_unknown_keys_pass_through_in_deobfuscate(opaque_map: SchemaMap):
    """Unknown opaque keys in deobfuscation are left unchanged."""
    doc = {
        opaque_map.field_name("title"): "Known field",
        "plaintext_key": "not obfuscated",
    }
    restored = opaque_map.deobfuscate_document(doc)

    assert "title" in restored
    assert restored["title"] == "Known field"
    assert "plaintext_key" in restored
    assert restored["plaintext_key"] == "not obfuscated"


# ── Properties ───────────────────────────────────────────────────────

def test_is_transparent_property():
    """is_transparent reflects the mode."""
    assert SchemaMap.transparent().is_transparent is True
    ns = UUID("12345678-1234-5678-1234-567812345678")
    assert SchemaMap(ns, mode="opaque").is_transparent is False


# ── StorageObfuscator protocol conformance ───────────────────────────

def test_schema_map_satisfies_storage_obfuscator_protocol():
    """SchemaMap must satisfy yanantin's StorageObfuscator protocol."""
    from yanantin.apacheta.storage_obfuscator import StorageObfuscator

    ns = UUID("12345678-1234-5678-1234-567812345678")
    schema_map = SchemaMap(ns, mode="opaque")
    assert isinstance(schema_map, StorageObfuscator)


def test_transparent_map_satisfies_protocol():
    """Transparent SchemaMap satisfies StorageObfuscator too."""
    from yanantin.apacheta.storage_obfuscator import StorageObfuscator

    schema_map = SchemaMap.transparent()
    assert isinstance(schema_map, StorageObfuscator)
