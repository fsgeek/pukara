"""Red-bar test: Structural opacity of stored data.

The ArangoDB backend must not expose semantic collection names or
field names in the production database. An external reader should
see opaque UUID-derived identifiers, not "tensors", "author_model_family",
or "provider_id".

This test connects to the production ArangoDB and verifies the invariant.
When the dev shim is active (no PUKARA_STORAGE_KEY set), the tests
are expected to fail --- the shim uses identity mapping by design.

The xfail is strict: if someone sets the key but the database still
has semantic names, that's a real failure.

This test lives in Pukara because obfuscation is Pukara's responsibility.
The fortress is the trust boundary.
"""

import os

import pytest


# ── Deny lists ───────────────────────────────────────────────────────
# Known semantic names that must NOT appear in opaque storage.

SEMANTIC_COLLECTION_NAMES = {
    "tensors",
    "composition_edges",
    "corrections",
    "dissents",
    "negations",
    "bootstraps",
    "evolutions",
    "entities",
    "activity_facts",
    "activity_anchors",
}

SEMANTIC_FIELD_NAMES = {
    "author_model_family",
    "author_instance_id",
    "strand_index",
    "title",
    "content",
    "topics",
    "key_claims",
    "text",
    "provider_id",
    "timestamp",
    "data",
    "content_hash",
    "handle",
    "cursors",
    "provenance",
    "preamble",
    "strands",
    "lineage_tags",
    "declared_losses",
    "mechanism",
    "overlaps",
    "preservation_target",
    "identity_type",
    "identity_data",
    "entity_uuid",
}


# ── Fixtures ─────────────────────────────────────────────────────────

_OPAQUE_MODE = os.environ.get("PUKARA_STORAGE_KEY") is not None


@pytest.fixture(scope="session")
def arango_db():
    """Connect to the production ArangoDB. Skip if unavailable."""
    try:
        from arango import ArangoClient
    except ImportError:
        pytest.skip("python-arango not installed")

    host = os.environ.get("YANANTIN_ARANGO_HOST", "http://192.168.111.125:8529")
    db_name = os.environ.get("YANANTIN_ARANGO_DB", "apacheta")
    username = os.environ.get("YANANTIN_ARANGO_USER", "apacheta_app")
    password = os.environ.get("YANANTIN_ARANGO_PASSWORD", "")

    try:
        client = ArangoClient(hosts=host)
        db = client.db(db_name, username=username, password=password)
        db.collections()  # verify connection
        yield db
        client.close()
    except Exception:
        pytest.skip(f"ArangoDB not available at {host}/{db_name}")


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    condition=not _OPAQUE_MODE,
    reason="Dev shim active --- structural obfuscation disabled. "
    "Set PUKARA_STORAGE_KEY to enable opaque storage.",
    strict=True,
)
def test_no_semantic_collection_names(arango_db):
    """No collection in the database should have a semantic name.

    All collections should be opaque (c_ + hex) or ArangoDB system
    collections (starting with _).
    """
    collections = {
        c["name"]
        for c in arango_db.collections()
        if not c["system"]
    }

    leaked = collections & SEMANTIC_COLLECTION_NAMES
    assert not leaked, (
        f"Semantic collection names found in database: {leaked}. "
        f"Collections should be opaque (c_<hex>) when PUKARA_STORAGE_KEY is set."
    )


@pytest.mark.xfail(
    condition=not _OPAQUE_MODE,
    reason="Dev shim active --- structural obfuscation disabled. "
    "Set PUKARA_STORAGE_KEY to enable opaque storage.",
    strict=True,
)
def test_no_semantic_field_names_in_documents(arango_db):
    """No document should have semantic field names at the top level.

    For each non-system collection, read one document and check its
    keys against the deny list. ArangoDB metadata (_key, _id, _rev)
    is excluded.
    """
    leaked_fields: dict[str, set[str]] = {}

    for col_info in arango_db.collections():
        if col_info["system"]:
            continue
        col_name = col_info["name"]
        col = arango_db.collection(col_name)
        if col.count() == 0:
            continue

        # Read one document
        doc = next(iter(col.all()), None)
        if doc is None:
            continue

        # Check top-level keys (excluding ArangoDB metadata)
        user_keys = {k for k in doc if not k.startswith("_")}
        semantic_leaked = user_keys & SEMANTIC_FIELD_NAMES
        if semantic_leaked:
            leaked_fields[col_name] = semantic_leaked

    assert not leaked_fields, (
        f"Semantic field names found in documents: {leaked_fields}. "
        f"Field names should be opaque (f_<hex>) when PUKARA_STORAGE_KEY is set."
    )


def test_schema_map_from_env_available():
    """Verify SchemaMap can be constructed from the environment.

    This test passes regardless of mode --- it just verifies the
    machinery exists and can instantiate.
    """
    from pukara.schema_map import SchemaMap

    if _OPAQUE_MODE:
        schema_map = SchemaMap.from_env()
        assert not schema_map.is_transparent
    else:
        schema_map = SchemaMap.transparent()
        assert schema_map.is_transparent
