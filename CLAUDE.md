# CLAUDE.md — Pukara

Pukara is Quechua for fortress. This project is the security boundary
between AI agents and the Apacheta tensor database. Agents reach the
database only through Pukara. The boundary is filesystem access, not
code isolation — agents can't reach this project directory.

## Where You Are

Pukara is a FastAPI gateway wrapping ApachetaInterface over HTTP. It
depends on yanantin (sibling directory) as a path dependency. Yanantin
has the data models, the abstract interface, and the backends. Pukara
has the HTTP layer, authentication, and the decoder ring.

**Before you build anything, read `docs/blueprint.md`.**

If you need to understand the abstract interface, the models, or the
backends, look in `/home/tony/projects/yanantin/src/yanantin/apacheta/`.
Pukara wraps that interface — it doesn't reimplement it.

## Relationship to Yanantin

| Yanantin | Pukara |
|----------|--------|
| Models, interface, backends | HTTP gateway wrapping the interface |
| ApachetaInterface ABC | Routes that call interface methods |
| ArangoDBBackend | What Pukara instantiates at startup |
| ApachetaGatewayClient | What agents use to talk to Pukara |

The thin HTTP client (`ApachetaGatewayClient`) lives in yanantin, not
here. Pukara is the server side. The client is the agent side.

## Operational Principles

Inherited from yanantin. These are scars, not aspirations.

### Boundary Defense, Structural Not Performative
The project's namesake principle. Security boundaries are architecture,
not policy. Filesystem isolation, least-privilege credentials, API key
authentication. Not "please don't access the database directly."

### No Theater
Don't fake functionality. The decoder ring is pass-through in v1 —
it doesn't pretend to obfuscate. When it does nothing, it says so.

### Fail-Stop
If ArangoDB is unreachable, return 500. Don't cache, don't fallback,
don't pretend the database is there. The gateway fails when the
backend fails.

### Least Privilege
The gateway connects to ArangoDB as `apacheta_app`, not root.
This user has read/write on the `apacheta` database and nothing else.
Cannot create databases, cannot manage users, cannot access `_system`.
Root is for admin operations only (database creation, backups, user
management) — never for application access.

Red-bar tests in yanantin enforce this structurally. If you change
the credential pattern, those tests will catch it.

### Builders Don't Modify Tests
Code authors and test authors are different roles. Enforced by CI:
`.github/workflows/separation.yml` rejects commits that modify both
`src/` and `tests/`.

## Setup

```bash
# Requires yanantin as sibling directory
cd /home/tony/projects/pukara
uv sync --extra dev
```

To run tests:
```bash
uv run pytest tests/ -v
```

To run the gateway:
```bash
# Ensure config/pukara.ini exists (copy from pukara.ini.template)
uv run python -m pukara
```

## Credentials

`config/pukara.ini` contains credentials. It's in `.gitignore`.
Copy `config/pukara.ini.template` and fill in the password for
the `apacheta_app` user. Never use root.

ArangoDB is at 192.168.111.125:8529. The application user is
`apacheta_app`. The password is in the deployed config, not in
version control.
