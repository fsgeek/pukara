# Pukara

**Pukara** (Quechua: *pukara*, "fortress") is a FastAPI gateway that provides
the sole access path between AI agents and the
[Apacheta](https://github.com/fsgeek/yanantin) tensor database.

## What It Does

Pukara wraps Yanantin's `ApachetaInterface` over HTTP, adding:

- **Boundary enforcement** — agents cannot reach ArangoDB directly; Pukara
  is the only entry point
- **Authentication** — API key validation (optional in development)
- **Schema obfuscation** — collection and field names map to opaque
  `c_<uuid5hex>` / `f_<uuid5hex>` identifiers under a per-deployment
  namespace key, so the stored schema is not self-documenting to the
  database provider. Identity mapping in dev (`PUKARA_STORAGE_KEY` unset);
  opaque in production. Content values are not encrypted — that's a
  declared loss, not a hidden one.
- **Audit logging** — every request logged with method, path, status, and
  elapsed time
- **Fail-stop error handling** — backend failures surface as HTTP errors,
  never silently swallowed

## Architecture

```
Agent (any AI model)
  → ApachetaGatewayClient (httpx, lives in yanantin)
    → Pukara (FastAPI, uvicorn)
      → SchemaMap (collection/field label obfuscation)
        → ArangoDBBackend (least-privilege credentials)
          → ArangoDB
```

Security is structural, not performative. The gateway uses a least-privilege
database user (`apacheta_app`) that can read and write the `apacheta`
database — nothing else. No root access, no system database access, no
database creation. Red-bar tests in Yanantin enforce these constraints
continuously.

## Part of Yanantin

Pukara is one component of a four-package project. [Tiksi](https://github.com/fsgeek/tiksi)
(Quechua: "foundation") holds the shared epistemic primitives — base
model, tensor records, provenance, composition edges — that everything
above it imports. [Yanantin](https://github.com/fsgeek/yanantin) (Quechua:
"complementary duality") wraps tiksi with the abstract storage interface
and the ArangoDB backend. Pukara provides the boundary defense.
[Willay](https://github.com/fsgeek/willay) (Quechua: "to tell, to inform")
produces epistemic receipts for claim-evidence verification, stored
through Pukara.

| Component | Role |
|-----------|------|
| **Tiksi** | Foundation models — `ApachetaBaseModel`, `TensorRecord`, provenance, composition |
| **Yanantin** | Storage interface and backends (ArangoDB) |
| **Pukara** | Gateway — the only door to the database |
| **Willay** | Epistemic receipts — verifiable attestation |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Tiksi and yanantin must be sibling directories
git clone https://github.com/fsgeek/tiksi.git
git clone https://github.com/fsgeek/yanantin.git
git clone https://github.com/fsgeek/pukara.git
cd pukara
cp config/pukara.ini.template config/pukara.ini
# Edit config/pukara.ini with your ArangoDB credentials —
# use the apacheta_app least-privilege user, never root.
uv sync
```

For opaque schema mode in production, set `PUKARA_STORAGE_KEY` to a UUID.
Unset means transparent (identity) mapping — fine for dev, not for
deployments where the database provider is in scope for the threat model.

Run the gateway:
```bash
uv run python -m pukara
```

Run tests:
```bash
uv run pytest tests/ -v
```

## API

Base URL: `http://127.0.0.1:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/version` | Gateway and interface versions |
| GET | `/counts` | Record counts per collection |
| POST | `/tensors` | Store a tensor record |
| GET | `/tensors` | List all tensors |
| GET | `/tensors/{id}` | Get a specific tensor |
| POST | `/composition-edges` | Store a composition edge |
| POST | `/corrections` | Store a correction record |
| POST | `/dissents` | Store a dissent record |
| POST | `/negations` | Store a negation record |
| GET | `/queries/...` | 20 query endpoints for epistemic, lineage, and provenance queries |

Full endpoint documentation available at `/docs` when the gateway is running.

## License

Open source. See [Yanantin](https://github.com/fsgeek/yanantin) for project
context and governance.
