# Pukara

**Pukara** (Quechua: *pukara*, "fortress") is a FastAPI gateway that provides
the sole access path between AI agents and the
[Apacheta](https://github.com/fsgeek/yanantin) tensor database.

## What It Does

Pukara wraps Yanantin's `ApachetaInterface` over HTTP, adding:

- **Boundary enforcement** — agents cannot reach ArangoDB directly; Pukara
  is the only entry point
- **Authentication** — API key validation (optional in development)
- **Decoder ring** — UUID obfuscation layer between agent-side and
  storage-side identifiers (v1: pass-through; v2: HMAC-based derivation)
- **Audit logging** — every request logged with method, path, status, and
  elapsed time
- **Fail-stop error handling** — backend failures surface as HTTP errors,
  never silently swallowed

## Architecture

```
Agent (any AI model)
  → ApachetaGatewayClient (httpx, lives in yanantin)
    → Pukara (FastAPI, uvicorn)
      → Decoder Ring (UUID mapping)
        → ArangoDBBackend (least-privilege credentials)
          → ArangoDB
```

Security is structural, not performative. The gateway uses a least-privilege
database user (`apacheta_app`) that can read and write the `apacheta`
database — nothing else. No root access, no system database access, no
database creation. Red-bar tests in Yanantin enforce these constraints
continuously.

## Part of Yanantin

Pukara is one component of the [Yanantin](https://github.com/fsgeek/yanantin)
project (Quechua: "complementary duality"). Yanantin provides the data
models, abstract interface, and storage backends. Pukara provides the
boundary defense. [Willay](https://github.com/fsgeek/willay) provides
epistemic receipts for claim-evidence verification, stored through Pukara.

| Component | Role |
|-----------|------|
| **Yanantin** | Tensor models, interface, backends |
| **Pukara** | Gateway — the only door to the database |
| **Willay** | Epistemic receipts — verifiable attestation |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Yanantin must be a sibling directory
git clone https://github.com/fsgeek/yanantin.git
git clone https://github.com/fsgeek/pukara.git
cd pukara
cp config/pukara.ini.template config/pukara.ini
# Edit config/pukara.ini with your ArangoDB credentials
uv sync
```

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
