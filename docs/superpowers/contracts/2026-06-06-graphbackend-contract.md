# GraphBackend Contract — yanantin-side work order

*Authored 2026-06-06 by the Pukara instance that owns the boundary.
The yanantin instance CONFORMS to this; it does not re-author it. If
something here is wrong or impossible against live code, raise it back
across the boundary — do not silently reshape the contract, because the
Pukara routes and the red-bar tests are written against exactly these
shapes.*

Design context: `pukara/docs/superpowers/specs/2026-06-06-llika-behind-pukara-design.md`.

**Line citations have a half-life. Re-verify every `file:line` below
against current code before editing.** (This contract was written
against code read 2026-06-06; the day's repeated lesson is that such
citations drift in days.)

---

## 1. `GraphBackend` capability (NOT on `ApachetaInterface`)

A narrow protocol, separate from the public `ApachetaInterface` domain
catalog. Do **not** add these verbs to `ApachetaInterface` — the find
spec forbids polluting that catalog (it "is already the leak"). Place the
protocol co-resident with the backend (e.g.
`yanantin/src/yanantin/apacheta/interface/graph.py` or alongside
`ArangoDBBackend`); exact module is yanantin's call, but it must be
importable by Pukara without dragging in the domain catalog.

`ArangoDBBackend` implements it. The implementation MUST route graph AQL
through the backend's existing obfuscator `self._map` (the same map that
obfuscates the record collections) — including the `llika_composition`
edge collection name and the traversal field paths. This closes the
latent plaintext bypass that exists today under the transparent default.

### Signatures (the wire contract — Pukara routes call exactly these)

```python
class GraphBackend(Protocol):
    def link(
        self,
        from_ref: str,           # "collection/<uuid>" vertex ref
        to_ref: str,             # "collection/<uuid>" vertex ref
        relation_type: RelationType,
        provenance: ProvenanceEnvelope,
        **fields,
    ) -> EdgeResult: ...

    def walk(
        self,
        start_id: str,           # "collection/<uuid>"
        direction: str,          # "forward" | "backward" | "both"
        depth: int,
        relation_types: list[str] | None = None,   # RelationType VALUES
        max_results: int = 50,
    ) -> list[PathResult]: ...

    def neighbors(
        self,
        start_id: str,
        direction: str,
        relation_types: list[str] | None = None,
    ) -> list[PathResult]: ...

    # get rides the EXISTING backend method — do NOT invent a new result
    # type. ApachetaInterface.get_record already returns a serializable
    # ApachetaBaseModel; Pukara serializes it model_dump(mode="json").
    # get_record(record_id: UUID) -> ApachetaBaseModel   (already exists)
```

`EdgeResult`, `PathStep`, `PathResult` are the existing frozen dataclasses
in `yanantin/src/yanantin/llika/models.py` — wire-safe already, reuse
verbatim. `RelationType` from `yanantin.apacheta.models.composition`.

**Open shape decision left to yanantin (flag the choice back):** `get`
takes a bare `UUID` (per `get_record`), but `walk`/`link` use
`"collection/<uuid>"` string refs. The id-shape seam (yanantin#10 SEAM 1)
wants the *public* contract to be bare UUIDs with slash-form converted at
the Arango boundary.

**FINDING (Pukara, 2026-06-06 — read before resolving): bare-UUID-
everywhere is WRONG against the data shape, and here is why.** The
vertices Llika links/traverses are NOT in one collection —
`_SEMANTIC_COLLECTIONS` (`arango.py:63`) holds `records`, `tensors`,
`entities`, … and a `link` edge's `_from`/`_to` can cross collections
(`tensors/<uuid>` → `records/<uuid>`). So a bare UUID into `link`/`walk`
is **ambiguous**: the backend cannot qualify `<uuid>` → `<collection>/
<uuid>` without knowing which collection the vertex lives in. `get` gets
away with bare UUID only because `get_record` looks solely in `records`
(`arango.py:308`). The graph verbs cannot.

Therefore the honest resolution is **MIXED, per-verb** (the ambiguity is
per-verb):
- `get` → bare `UUID` (records-only, unambiguous).
- `link` / `walk` / `neighbors` → `"collection/<uuid>"` slash-form
  (cross-collection, the qualifier is intrinsic to the ref).

**This was NOT pinned unilaterally** because it has a cross-repo cost the
Pukara instance could not see: yanantin#10 SEAM 1 says Hamut'ay's
`tool_recall` parses bare `UUID(...)` and RAISES on slash-form. Resolving
mixed-shape means Hamut'ay must either (a) only ever address records via
`get` (bare, fine), or (b) learn to handle slash-form for graph refs. The
deciding fact — does Hamut'ay's traversal need cross-collection edge refs,
or is it records-only? — was not visible to the Pukara hand. **Resolve as
mixed-shape UNLESS Hamut'ay's needs are known to be records-only, in which
case bare-UUID-everywhere becomes safe.** Whatever you pick, `get` and
`walk` must present a coherent surface and the gateway client must match.
State the chosen convention back.

NOTE: the current Pukara `routes/llika.py` types `from_ref`/`start_id` as
`str` (slash-form), consistent with the mixed resolution above. If the
decision lands on bare-UUID-everywhere, the routes change too.

---

## 2. `ProvenanceEnvelope` discriminator (tiksi-side)

Add ONE field to `ProvenanceEnvelope` in `tiksi/src/tiksi/provenance.py`
(currently the class at `:21`, fields end with `interface_version: str =
"v1"`):

```python
authorship_verified: bool = False
```

**Default MUST be `False`.** Rationale is load-bearing: every existing
record and every gateway-written edge is then honestly marked *unverified*
by default. Verification must be *earned* (something proves identity and
flips it to `True`); it can never be accidentally inherited. Until
yanantin#13 (identity) lands, nothing flips it — so every edge written
through Pukara carries `authorship_verified=False` in the stored data,
where a future reader sees it.

Do not add validation that rejects `True` — a later identity subsystem
needs to set it. Just the field, defaulted false. Tiksi is published to
PyPI; this is a backward-compatible additive field (new optional field
with a default), so existing serialized records deserialize unchanged
(absent → `False`).

`tiksi` change ripples to yanantin's re-export shim and any
`ProvenanceEnvelope(...)` construction sites — verify they still
construct (they will; the field has a default).

---

## 3. `LlikaService` becomes a facade — DELETE the raw handle

`yanantin/src/yanantin/llika/service.py`. Today `__init__` calls
`ApachetaDBConfig().connect(tier)` (verified `service.py:40`) and holds a
privileged `StandardDatabase`. 

**Mandate: delete `connect(tier)`. Not deprecate. Delete.** The facade
holds a `GraphBackend`, not a DB handle. A commented-out or
feature-flagged raw handle FAILS the intent — erosion routes through
comments. Red-bar test #1 (Codex) is written to go red if any
agent-reachable path can obtain a raw handle.

The facade's verbs (`link`/`walk`/`neighbors`, plus a `get` delegating to
`get_record`) become thin pass-throughs to the `GraphBackend`. Result
types unchanged. `link`/`walk` AQL that currently lives in the service
moves into the `ArangoDBBackend` `GraphBackend` implementation (where it
gets obfuscator routing for free).

The `tier` parameter: note in passing it was conflating
deployment-environment with tenant (yanantin#13). This slice does NOT
resolve that — `tier` may remain as the env selector for now — but do not
add new `tier`-as-tenant semantics. Identity is #13's job.

---

## 4. `ApachetaGatewayClient` — four verbs over httpx

`yanantin/src/yanantin/apacheta/clients/gateway.py`. Add `link`, `walk`,
`neighbors`, `get` mirroring the Pukara routes (Section 5) over httpx,
following the existing client's HTTP→`ApachetaError` ladder. The client
is the agent-side surface; its method shapes must match the route
contract exactly. Build these AFTER the Pukara routes exist (Section 5)
so you mirror the real endpoints.

---

## 5. What Pukara builds (for your reference — do NOT build this side)

`pukara/src/pukara/routes/llika.py`,
`APIRouter(prefix="/api/v1/llika")`, on `Depends(get_backend)`:

| Route | Method | Calls |
|---|---|---|
| `/api/v1/llika/link` | POST | `backend.link(from_ref, to_ref, relation_type, provenance, **fields)` |
| `/api/v1/llika/walk` | POST | `backend.walk(start_id, direction, depth, relation_types, max_results)` |
| `/api/v1/llika/neighbors` | POST | `backend.neighbors(start_id, direction, relation_types)` |
| `/api/v1/llika/get/{record_id}` | GET | `backend.get_record(record_id)` → `model_dump(mode="json")` |

`link`'s request body carries the agent-supplied `ProvenanceEnvelope`
verbatim (transport trust — see spec §Provenance). Pukara does NOT
synthesize or verify it.

`get_backend` returns `app.state.backend` (the `ArangoDBBackend`), which
must satisfy `GraphBackend`. The Pukara routes depend only on the
`GraphBackend` surface + `get_record`.

---

## 6. Conformance checklist (yanantin instance ticks these)

- [ ] `GraphBackend` protocol defined off the public `ApachetaInterface` catalog.
- [ ] `ArangoDBBackend` implements it; graph AQL routes through `self._map`.
- [ ] `link`/`walk`/`neighbors` signatures match Section 1 exactly.
- [ ] `get` rides existing `get_record`; no new result type invented.
- [ ] id-shape convention chosen and stated back; `get`/`walk` agree at the boundary.
- [ ] `ProvenanceEnvelope.authorship_verified: bool = False` added (tiksi); default false; no rejection of true.
- [ ] `LlikaService.connect(tier)` raw handle DELETED; facade holds `GraphBackend`.
- [ ] `ApachetaGatewayClient` gains link/walk/neighbors/get matching the routes.
- [ ] Both repos: builder commits (`src/`) separate from any tester commits (`tests/`).
- [ ] Anything wrong/impossible here raised back, not silently reshaped.
