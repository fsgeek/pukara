# Llika Slice — Red-Bar Test Specs (for Codex)

*Authored 2026-06-06 by the Pukara instance. These are SPECS. Codex
authors the actual tests, blind to both builders, so neither the Pukara
builder nor the yanantin builder can write the wall to pass. The wall is
the deliverable; the routes are plumbing.*

Design: `2026-06-06-llika-behind-pukara-design.md`.
Contract: `2026-06-06-graphbackend-contract.md`.

## Sequencing — expected red is not failure

Some of these red bars are MEANT to be red until the yanantin-side build
lands (GraphBackend, `connect(tier)` deleted, `authorship_verified`
added). A red bar that goes green when the wall is built is the design
working. Each spec below is tagged:

- **[RUNS NOW]** — testable against the current tree (Pukara routes exist).
- **[RED UNTIL YANANTIN]** — depends on the yanantin-side build; expected
  red now, must go green when that lands. Do not "make it pass" by
  weakening it — its redness is load-bearing until the wall exists.

State clearly in each test's docstring which it is, and reference
yanantin#13 where the hole it guards is the identity gap.

---

## Test 1 — Gateway is the only door  [RED UNTIL YANANTIN]

**Guards:** the wall the builder trips over. Fails if any agent-reachable
code can obtain a raw privileged DB handle / reach Arango outside Pukara.

**Assertion:** `LlikaService` (after the facade refactor) does NOT hold or
construct a `StandardDatabase` / call `ApachetaDBConfig().connect(...)`.
Concretely: constructing a `LlikaService` must not open a privileged DB
connection of its own; its only path to data is the `GraphBackend` it is
given.

**Why red now:** today `LlikaService.__init__` still calls
`connect(tier)` (`yanantin/.../llika/service.py:40`). This test is red
until that deletion lands. It must go red again the day any future
instance re-adds a raw handle — that is its whole purpose.

**Probe ideas (Codex, extend freely):** inspect that `LlikaService` has no
attribute holding a `StandardDatabase`; assert `connect` is not called
during construction (e.g. patch `ApachetaDBConfig.connect` to raise and
assert it is never hit); grep-style structural assertion that
`connect(` does not appear in the service module. Prefer a behavioral
assertion over a string-grep where possible, but a string-grep tripwire
on `connect(tier)` is acceptable as a backstop — name it as such.

---

## Test 2 — Stored record carries the unverified mark  [RED UNTIL YANANTIN]

**Guards:** the DATA's honesty, not merely the gateway's behavior. This is
Yanantin review change 1: the mark must live where the future reader is
(in the edge), not only where the builder is (in the test).

**Assertion:** an edge written through `POST /api/v1/llika/link` is
*stored* with `provenance.authorship_verified == False`. Not "the gateway
accepted the write" — the persisted record carries the unverified mark.

**Why red now:** `ProvenanceEnvelope` has no `authorship_verified` field
yet (added tiksi-side per the contract), and the backend `link` does not
exist yet. Red until both land.

**Critical — do NOT weaken to the gateway-accepted version.** A test that
only asserts "Pukara returned 201" documents the hole to the builder and
leaves it undocumented to the future query that trusts the data. The
assertion must read the STORED record (round-trip: link, then fetch the
edge / get the record) and check the mark is present and False. See
yanantin#13: until identity lands, nothing may flip it to True.

---

## Test 3 — `check_access` is a no-op tripwire  [RUNS NOW]

**Guards:** Hole 2 — authorization is currently a stub that always allows.

**Assertion:** `ApachetaInterface.check_access(caller, operation, target)`
returns `True` unconditionally today (`yanantin/.../interface/abstract.py`
~:45, "Always returns True in v1"). The test pins this as a KNOWN,
DOCUMENTED no-op — so the day someone wires real enforcement, the test
flips meaning loudly rather than the stub silently passing.

**Why runs now:** the stub exists today; the test documents current
reality and references yanantin#13. When enforcement lands, this test is
the one that must be deliberately rewritten — that is the signal.

**Shape:** assert `check_access(...)` returns True for a caller/op that a
real authorization layer would plausibly DENY (e.g. an unknown caller
attempting a write). The test's docstring states: "This passing is the
HOLE, not the feature. yanantin#13."

---

## Test 4 — `@aid` filter is convenience, not boundary  [RUNS NOW]

**Guards:** Hole 3 — `author_instance_id` filtering is a caller-supplied
query param, not an isolation boundary.

**Assertion:** any caller can pass any `author_instance_id` to
`query_open_by_author_instance` (`arango.py` ~:677) and read records
labeled with that id — there is no check that the caller IS that author.
Demonstrate that caller B can read records authored (labeled) by A by
simply passing A's id.

**Why runs now:** the filter exists today and takes a caller-supplied
`@aid`. The test pins it as a query feature, forbidding its mistaken use
as tenant isolation. Docstring: "This is a query convenience. It is NOT
access control. yanantin#13."

---

## Separation

These are TESTER commits (`tests/`), authored by Codex, separate from any
builder commit. Whether they land in pukara's `tests/` or yanantin's
`tests/red_bar/` depends on what each asserts: tests 1–2 and the
behavioral parts of 3–4 that exercise the backend likely belong in
yanantin's red-bar suite (they assert on yanantin/tiksi internals); the
gateway round-trip in test 2 needs Pukara's TestClient. Codex decides
placement per assertion, keeping each test in the repo whose code it
guards, and never mixing `src/` and `tests/` in one commit in either repo.
