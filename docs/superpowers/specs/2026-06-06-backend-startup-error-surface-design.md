# Surface Backend Startup Failure at the Pukara Boundary

*Design — 2026-06-06. Settled via brainstorming with Tony.*

## Context

Yanantin shipped discriminated backend connection errors (commit history
in yanantin; types in `yanantin/apacheta/interface/errors.py`):

- `BackendAuthError` — credentials rejected (HTTP 401/403)
- `BackendUnreachableError` — host never answered (transport failure)
- `DatabaseNotProvisionedError` — reachable, authenticated, db missing

All three subclass the builtin `ConnectionError`, so existing
`except ConnectionError` sites keep working while new code can branch on
the specific cause. `ArangoDBBackend._discriminate_connection_failure()`
produces them with remediation-honest messages (e.g. auth failures say
"check credentials, not a provisioning problem").

This is the Pukara-side follow-up the pointer doc
(`docs/plans/2026-05-19-backend-error-pointer.md`) parked, gated on the
yanantin discrimination landing. It has landed and is verified live.

## Decision (settled)

**Fail-stop stays.** Tony's call, on distributed-systems grounds: a
degraded mode that comes up and returns 503 is a liveness lie wearing a
503 costume — FLP-flavored. If the backend isn't there, the gateway isn't
there. CLAUDE.md's fail-stop principle is honored literally, not just in
spirit.

The *only* gap: today the discriminated diagnosis dies in an anonymous
traceback out of `lifespan`. The three exception types yanantin built are
wasted because Pukara catches none of them. This change makes the
diagnosis **visible** at the boundary before fail-stop proceeds.

Two sub-decisions:

- **Noise and observability.** Log the honest one-liner *and* keep the
  full stack. Nothing suppressed. ("No theater" applied to failure.)
- **Pass through, don't re-interpret.** Pukara logs `str(exc)` plus the
  class name. Yanantin owns the *wording* of the diagnosis; Pukara owns
  its *visibility*. No per-class remediation branching in Pukara —
  "premature optimization is the special case of premature collapse." If
  per-class remediation justifies itself later, branch then.

## Change

One `try/except` around backend construction in `src/pukara/app.py`
`lifespan`:

```python
try:
    backend = ArangoDBBackend(...)
except ConnectionError as exc:
    logger.error("Backend startup failed [%s]: %s", type(exc).__name__, exc)
    raise
```

- Catch `ConnectionError` — common base of all three discriminated types,
  and the old blanket `ConnectionError` if anything still raises it.
- `bare raise` re-raises the *same* exception with its chained stack
  (yanantin already does `raise ... from e`). uvicorn aborts startup;
  process exits non-zero.
- Catch `ConnectionError`, **not** `Exception`. A bug in Pukara's own
  startup (e.g. a bad `UUID(config.storage_key)`) must propagate raw,
  *not* be mislabeled "Backend startup failed."

Nothing else in `src/` changes. `/health` stays static: under fail-stop
the process is dead, so there is no degraded-state surface to report.

## Testing

Authored by **Codex**, not Claude — this both satisfies builder/tester
separation (different hand) and leans on Codex's edge-case nose, per
Tony's standing observation that it finds things Claude models miss.

New file `tests/test_startup_failure.py`. Minimum assertions:

1. Startup against an unreachable/unauthenticated backend raises a
   `ConnectionError` subclass — fail-stop holds (likely via `TestClient`
   context-manager entry, which triggers lifespan).
2. The boundary log line is emitted with the exception class name
   (`caplog`).

Codex is invited to extend beyond the minimum — e.g. does each of the
three discriminated types surface correctly through the boundary? Does a
non-`ConnectionError` startup bug propagate *un*labeled, as designed?

## Commits

Two, to respect `.github/workflows/separation.yml` (per-commit rule, no
single commit mixes `src/` and `tests/`):

1. **Builder (Claude):** `src/pukara/app.py` — the try/except boundary.
2. **Tester (Codex):** `tests/test_startup_failure.py`.

## Related

- `docs/plans/2026-05-19-backend-error-pointer.md` — the parked pointer
  this resolves.
- `yanantin/docs/plans/2026-05-19-backend-connection-error-discrimination.md`
  — canonical upstream plan.
- `memory/feedback_threat_model_default.md` — the meta-pattern (adversary
  is the database, not the caller).
