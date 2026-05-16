# Blueprint — Pukara

*Not a tensor. A map of what exists and what doesn't.*

*Last updated: 2026-05-16*

## What Exists

### Gateway (src/pukara/)

12 Python files, 2 classes, 36 HTTP endpoints.

| File | What it does |
|------|-------------|
| `app.py` | Application factory. Lifespan creates ArangoDBBackend with a `SchemaMap` (opaque if `PUKARA_STORAGE_KEY` set, else transparent), registers exception handlers (409/404/403/400/500), audit logging middleware. |
| `config.py` | `PukaraConfig` dataclass. Reads INI file + env var overrides. Defaults are inert (`localhost`, `apacheta_app`, blank password) — config file or env vars must supply real values. |
| `auth.py` | API key authentication. Empty key = dev mode (no auth). |
| `schema_map.py` | `SchemaMap` — collection and field label obfuscation. Maps semantic names to `c_<uuid5hex>` / `f_<uuid5hex>` under a per-deployment UUID namespace. Transparent mode = identity map for dev. Content values are not encrypted (declared loss). |
| `deps.py` | FastAPI dependency injection: `get_backend()`. |
| `__main__.py` | `python -m pukara` entry point. |
| `routes/store.py` | POST endpoints — one per record type. Return 201. |
| `routes/read.py` | GET endpoints — list tensors, get tensor, get strand, get entity. |
| `routes/query.py` | GET query endpoints across the epistemic, lineage, and provenance categories. |
| `routes/meta.py` | health, version, counts. |

### Tests (tests/)

| File | Tests | Author |
|------|-------|--------|
| `test_gateway.py` | 15 | Builder (T12 session — violation acknowledged) |
| `test_gateway_independent.py` | 131 | Independent Sonnet agent |
| **Total** | **146** | |

### CI (.github/workflows/separation.yml)

Two jobs:
1. **check-separation**: Rejects commits modifying both `src/` and `tests/`
2. **tests**: Runs full test suite (checks out both pukara and yanantin)

### Config

- `config/pukara.ini.template` — checked in, shows least-privilege pattern
- `config/pukara.ini` — gitignored, contains actual credentials
- `.gitignore` — covers config, __pycache__, .venv, etc.

## What Connects

```
Agent (any AI instance)
  │
  │ ApachetaGatewayClient (lives in yanantin, not here)
  │   ↓ HTTP
  │
Pukara (this project)
  │ X-API-Key header (auth.py)
  │ SchemaMap (schema_map.py) — opaque or transparent per PUKARA_STORAGE_KEY
  │ Audit log (app.py middleware)
  │   ↓ Python import
  │
ArangoDBBackend (lives in yanantin)
  │ apacheta_app user (least privilege, rw on apacheta db only)
  │   ↓ python-arango
  │
ArangoDB (192.168.111.125:8529)
  │ apacheta database
  │ 8 collections: tensors, composition_edges, corrections,
  │   dissents, negations, bootstraps, evolutions, entities
```

## What Doesn't Exist

| Gap | What it would be |
|-----|-----------------|
| **Content obfuscation** | `SchemaMap` hides collection and field names, not values. A reader of the raw documents still sees the data. Real content protection requires designing around ArangoDB's indexing constraint — order-preserving encryption, application-level encrypt/decrypt, or encrypted search. Declared loss, not a hidden one. |
| **Identifier obfuscation** | The original "decoder ring" idea — translating agent-side record UUIDs to storage-side UUIDs so a DB-side adversary can't correlate agent-known IDs with stored rows. Not implemented. Different threat surface from `SchemaMap`, which addresses schema-shape inference. |
| **Red-bar tests** | Pukara has no structural invariant tests. Yanantin has them for least-privilege, immutability, provenance, monotonicity. Pukara should have them for: gateway is the only entry point, `SchemaMap` is applied to every collection/field crossing the boundary, no direct backend access exposed. |
| **Rate limiting** | No request rate limiting. Dev mode has no auth at all. |
| **Audit persistence** | Audit log goes to stdout. Not persisted, not queryable. |
| **HTTPS** | HTTP only. TLS termination assumed to be handled by reverse proxy in production. |

## ArangoDB Users

| User | Access | Purpose |
|------|--------|---------|
| `root` | Everything | Admin only: create databases, create users, backups. Never for application access. |
| `apacheta_app` | rw on `apacheta` | Gateway application access. What Pukara uses. |
| `apacheta_test` | rw on `apacheta_test` | Integration test access. Used by yanantin's test suite. |

## How to Update This Blueprint

This document describes what IS. When you build something, update it.
When something here becomes wrong, fix it. A blueprint that doesn't
match the building is worse than no blueprint.
