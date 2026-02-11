# Blueprint — Pukara

*Not a tensor. A map of what exists and what doesn't.*

*Last updated: T13 session, 2026-02-10*

## What Exists

### Gateway (src/pukara/)

12 Python files, 2 classes, 39 HTTP endpoints.

| File | What it does |
|------|-------------|
| `app.py` | Application factory. Lifespan creates ArangoDBBackend, registers exception handlers (409/404/403/400/500), audit logging middleware. |
| `config.py` | `PukaraConfig` dataclass. Reads INI file + env var overrides. |
| `auth.py` | API key authentication. Empty key = dev mode (no auth). |
| `decoder.py` | `DecoderRing` — UUID obfuscation between agents and storage. V1 = pass-through. |
| `deps.py` | FastAPI dependency injection: `get_backend()`, `get_decoder()`. |
| `__main__.py` | `python -m pukara` entry point. |
| `routes/store.py` | 8 POST endpoints — one per record type. Return 201. |
| `routes/read.py` | 4 GET endpoints — list tensors, get tensor, get strand, get entity. |
| `routes/query.py` | 20 GET query endpoints across 7 categories. |
| `routes/meta.py` | health, version, counts. |

### Tests (tests/)

| File | Tests | Author |
|------|-------|--------|
| `test_gateway.py` | 15 | Builder (T12 session — violation acknowledged) |
| `test_gateway_independent.py` | 135 | Independent Sonnet agent |
| **Total** | **150** | |

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
  │ DecoderRing (decoder.py) — pass-through v1
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
| **DecoderRing v2** | Real UUID obfuscation. HMAC-based derivation so storage can't correlate agent UUIDs with stored UUIDs. The "Kraken poo" problem — anyone with DB access can read labels. |
| **Red-bar tests** | Pukara has no structural invariant tests. Yanantin has them for least-privilege, immutability, provenance, monotonicity. Pukara should have them for: gateway is the only entry point, decoder ring is applied to all UUIDs, no direct backend access exposed. |
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
