# Llika Behind Pukara — Slice 1 of yanantin#10

*Design — 2026-06-06. Brainstormed by the Pukara instance, reviewed by
Yanantin, two changes folded in. Three-hand build.*

## Scope

Slice 1 of yanantin#10 only: route Llika's graph surface through the
Pukara gateway so `LlikaService` stops holding its own privileged
ArangoDB handle. **Deferred to later slices:** value-obfuscation
(SchemaMap value-mapping) and the plaintext-adversary validator. Those
depend on this slice (the obfuscator lives Pukara-side; Llika must route
through Pukara before value-obf is reachable) — this slice is their
precondition, not their peer.

## The problem (verified against live code 2026-06-06)

`LlikaService.__init__(tier, provenance)` calls
`ApachetaDBConfig().connect(tier)` — it resolves its **own** privileged
`apacheta_app` ArangoDB handle inside agent-reachable code, and even
creates collections. Zero `llika` refs exist in `pukara/`. This is not a
breach of an existing wall — **the wall was never built on this side.**
`LlikaService` was written in a world where the Llika gateway route did
not exist; it reached for the raw handle because that was the only door.

Remediation is therefore *unwalled-gate*, not breach-repair: build the
missing wall and make a red-bar test the precondition for the gate ever
opening. The low-friction bypass (`connect(tier)`) is what erodes
boundaries between sessions; the wall must trip on *that*, not on a
hypothetical external attacker.

## Surface

Four verbs: **`link`, `walk`, `neighbors`, `get`.**

**NOT `find`.** `find` is intentionally absent from `LlikaService` (its
docstring: "a callable predicate cannot cross a wire; the customer
filters by structure"). The future declarative `find` is later
yanantin-side work gated *behind* this boundary, not part of it. (This
corrected a stale "walk/neighbors/find/get" list the agenda carried.)

`get` is new — `LlikaService` today has no single-record read. It rides
the backend's existing `get_record(record_id) -> ApachetaBaseModel`
(already on `ApachetaInterface`), serialized `model_dump(mode="json")`.
No new result type is invented for it.

## Architecture — option (c): narrow graph capability, off the public catalog

The find spec explicitly forbids adding graph verbs to the public
`ApachetaInterface` domain catalog: that catalog "is already the leak"
(spec lines 53–64, 493, 660). So agenda-option-(a) (verbs on
`ApachetaInterface`) is rejected — it pollutes the ~40-method domain
catalog with graph semantics.

Instead: a **narrow `GraphBackend` capability** co-resident with
`ArangoDBBackend`, separate from the public catalog.

- **Yanantin side (built by a separate yanantin instance — see Build):**
  - `GraphBackend` protocol: `link` / `walk` / `neighbors` / `get_record`.
  - `ArangoDBBackend` implements it (it already owns the obfuscator `_map`,
    so graph AQL routes through the obfuscator — closing the latent
    `llika_composition` plaintext bypass under the transparent default).
  - `LlikaService` becomes a **thin facade** over the backend capability
    and **loses its `connect(tier)` handle entirely.**
  - `ApachetaGatewayClient` grows the four verbs over httpx.
- **Pukara side (built by this instance):**
  - `routes/llika.py` (`APIRouter(prefix="/api/v1/llika")`) on the
    existing `Depends(get_backend)` pattern.

**Delete, don't deprecate.** `connect(tier)` is removed, not commented
out. A deprecated raw handle is a bypass with a comment on it, and
erosion routes straight through comments. (Heracles rerouted the river;
he did not leave the old channel with a sign.)

Data flow: `agent → ApachetaGatewayClient → HTTP → routes/llika.py →
get_backend → GraphBackend(ArangoDBBackend) → ArangoDB`.

## Provenance — transport trust, not truth trust (Yanantin change 1)

A `link` write through the gateway carries a `ProvenanceEnvelope` the
**agent supplies in the request body**; Pukara passes it through
verbatim. Pukara is *not* the author and (today) has no caller identity
to derive authorship from — its auth is a single anonymous shared API
key. So the provenance is a **claim, not a fact.**

The word "trusted" here means **transport trust** (Pukara faithfully
records what it was told), NOT **truth trust** (Pukara vouches it). The
unsafe reading — accept and write the provenance full stop, so a future
query reads `author_instance_id: X` and *believes* it — must be
structurally prevented.

**The mark lives in the data, not only in the test.** Add a discriminator
to `ProvenanceEnvelope` (tiksi, `src/tiksi/provenance.py`):

```python
authorship_verified: bool = False
```

Default `False` — every record is honestly marked unverified unless
something *proves* verification and flips it. You cannot accidentally
inherit `True`; verification must be earned. Until yanantin#13 (identity)
lands, nothing can flip it, so every gateway-written edge carries
`authorship_verified=False` in the stored record. A reader two years out
querying the edge sees the mark in the data — they do not run the test
suite first. This is the production/consumption invariant at the storage
layer: "Pukara did not verify this" is a production-time certainty that
must be in the edge.

## Single-tenant-but-not-foreclosed (yanantin#13)

This slice ships on a substrate that **cannot attribute memory to a
distinct instance** — caller identity is asserted, never verified, at
three layers (anonymous auth; `check_access` → `True`; `@aid` filter is a
caller-supplied query param, not an isolation boundary). Filed as
yanantin#13. It does **not** block this slice: identity enforces *at the
gateway*, and this slice builds the gateway — fixing identity first is
building the lock before the door.

The three holes are not hidden; they are turned into **named,
test-guarded red bars** (see Testing). The identity subsystem is later
work, built on **evidence from the single-instance frontier, not guessed
from human user-identity shapes** (those were half-built here and already
the wrong shape).

## Testing — the four red bars are the center of gravity

The wall is the deliverable; the routes are plumbing. All four authored
by **Codex**, blind to both builders (so neither can write the wall to
pass). All reference yanantin#13 where relevant.

1. **Gateway-is-the-only-door.** Fails if `LlikaService` (or any
   agent-reachable code) can obtain a raw DB handle / reach Arango outside
   Pukara — i.e. fails the day a future instance re-adds `connect(tier)`.
   This is the wall the builder trips over.
2. **Hole 1 — stored record carries the unverified mark.** Asserts a
   gateway-written `link` edge is *stored* with `authorship_verified ==
   False`. Guards the **data's** honesty, not only that the gateway
   accepted the write. Goes red when #13 lands and the verified path must
   set it.
3. **Hole 2 — `check_access` is a no-op tripwire.** Fails meaningfully
   when `check_access`'s `return True` is wired to real enforcement, so
   the guard flips correctly rather than silently.
4. **Hole 3 — `@aid` filter is convenience-not-boundary.** Demonstrates
   any caller can pass any `author_instance_id` and read those records.
   Pins it as a query feature, forbidding its mistaken use as isolation.

## Build — three hands, each justified by the act it protects

Not three-for-virtue. The one irreducible act the wall constrains —
**deleting `connect(tier)` and building the facade** — must not be done
by the instance that *defines* the boundary, or the wall is built in the
same head as the thing it constrains (the bypass-builder-is-the-fix-
builder problem, one level down). The wall test must be written by
neither builder.

- **This Pukara instance:** authors the contract (GraphBackend protocol,
  wire signatures, the discriminator requirement, the four red-bar specs)
  and builds the Pukara side (`routes/llika.py`). Contract is *authored*
  here because Pukara owns the boundary; it is not *satisfied* here.
- **A separate yanantin instance:** builds `GraphBackend`, the
  `LlikaService` facade, the tiksi discriminator, and performs the
  `connect(tier)` deletion — *conforming* to the contract, not authoring
  it.
- **Codex:** authors the four red-bar tests, blind to both.

Everything else collapses toward fewer hands. The separation is spent
only where a specific act demands it.

The contract this instance hands to the yanantin instance lives in
`docs/superpowers/contracts/2026-06-06-graphbackend-contract.md`.

## Cross-repo / separation

Spans pukara + yanantin + tiksi. Both pukara and yanantin enforce
builder/tester separation per-commit (`src/` and `tests/` never in one
commit). The Pukara-side build is a builder commit (`src/` only); the
red-bar tests are a separate Codex tester commit.

## Error handling

Reuse Pukara's existing exception-handler ladder (`ApachetaError`→500,
`NotFoundError`→404, `AccessDeniedError`→403, etc.). A `walk`/`get` on a
missing start vertex → `NotFoundError`→404. No new handlers unless a
graph verb raises something unmapped — flagged to the tester as a TBD to
probe rather than pre-solved.

## Related

- yanantin#10 (canonical agenda), yanantin#13 (identity blocker)
- `memory/project_multitenancy_absent_verified.md`
- find spec: `yanantin/docs/superpowers/specs/2026-06-02-llika-find-goal-focused-recall-design.md`
- `memory/feedback_threat_model_default.md`
