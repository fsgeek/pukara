# Backend Connection Error Discrimination — Pointer

*Captured 2026-05-19 — cross-reference only. Canonical plan is upstream.*

The Indaleko misdiagnosis pattern (privilege errors framed as
connectivity errors) was verified live in `ArangoDBBackend`. The
canonical fix plan lives in yanantin:

`/home/tony/projects/yanantin/docs/plans/2026-05-19-backend-connection-error-discrimination.md`

## What This Means for Pukara

The fix is yanantin-side. Pukara inherits the corrected exception
discrimination automatically by virtue of the path dependency. But
there is one Pukara-side decision the upstream plan defers to us:

**When the backend fails to connect during lifespan startup, should
Pukara fail-stop (current behavior) or come up in degraded mode with
each route returning a 503 derived from the specific error class?**

- *Current (fail-stop):* uvicorn fails to bind, agent sees connection
  refused, no diagnostic surface.
- *Degraded mode:* app comes up, routes return 503 with the
  discriminated error (auth vs. unreachable vs. provisioning),
  `/health` reports backend state. Agent has signal.

CLAUDE.md's fail-stop principle says "if ArangoDB is unreachable,
return 500. Don't cache, don't fallback, don't pretend the database
is there." Degraded mode arguably violates this. But the *intent* of
fail-stop is "don't fake a working database," not "give the agent no
information." Degraded mode preserves the intent (every request still
fails) while restoring the diagnostic surface.

This decision should happen *after* yanantin's discrimination is
implemented — without typed exceptions, Pukara cannot meaningfully
discriminate in its 503 detail anyway.

## RESOLVED 2026-06-06

Yanantin's discrimination landed (`BackendAuthError`,
`BackendUnreachableError`, `DatabaseNotProvisionedError`, all
`ConnectionError` subclasses). Pukara's decision: **fail-stop, not
degraded mode** (FLP grounds — a degraded app returning 503 is a
liveness lie wearing a 503 costume). The gap was that the discriminated
diagnosis died in an anonymous traceback; fixed by catching
`ConnectionError` at the `lifespan` boundary, logging the honest message
+ class name, then re-raising. Fail-stop intact.

Spec: `docs/superpowers/specs/2026-06-06-backend-startup-error-surface-design.md`
Commits: builder `1de60a3`, tester `3f2cfd0` (Codex-authored).

## Related

- `pukara/memory/feedback_threat_model_default.md`
- `pukara/docs/plans/2026-05-16-schema-extras-and-registration.md`
