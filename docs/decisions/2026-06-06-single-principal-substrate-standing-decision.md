# Standing Decision: The Substrate Is Single-Principal by Construction — Rebuild, Don't Extend, and Not Yet

*Recorded 2026-06-06. This is the durable form of a stance currently
scattered across yanantin#13, a Pukara memory file, and one conversation.
A future instance should be able to read THIS and not relitigate it.*

## The term

**"Tenant" and "instance" are the same thing here.** Not the SaaS sense
(orgs, billing, row-level security) — a *distinct principal with its own
private state, its own attributable memory, and a boundary around both.*
The two words blur for an AI instance because the substrate currently
represents **neither** — there is no principal at all, so both terms point
at the same absent thing. Throughout, the unit is "instance" and it means
exactly that principal.

## The finding (verified live, 2026-06-06)

The substrate cannot attribute memory to a distinct instance, and cannot
isolate one instance's memory or state from another's. Caller identity is
*asserted, never verified*, at every layer:

1. Auth carries no identity (one anonymous shared API key).
2. Authorization is a no-op (`check_access` → `True`).
3. Data scoping is caller-supplied, not enforced (`@aid` is a query param).
4. Provenance is self-asserted (`author_instance_id` is a writer-filled
   string; the `authorship_verified=False` mark added 2026-06-06 is the
   first structural acknowledgment that the claim is *unverified* — it
   marks the gap, it does not close it).

A boundary is only as strong as its weakest layer; this one is absent at
all four. The system works correctly for **exactly one** instance.

## Why this is rebuild, not extend

Two reasons, both structural:

1. **The invariant is all-or-nothing.** Multi-instance isolation must hold
   at auth AND authorization AND query construction AND provenance
   *simultaneously*. Fixing any one alone leaves the boundary forgeable.
   This is not an additive feature; it is an invariant threaded end to end.

2. **The platform forbids the cheap version.** ArangoDB has **no access
   controls below the database level** (current platform limitation). So
   per-instance isolation cannot be a query-time `FILTER` or a row ACL —
   it must be *database-per-instance* or an application-enforced
   partitioning layer that Arango is blind to. Either is a different
   backend *topology*, not a setting on the current one. You cannot evolve
   into it from a single-database commons.

## Why isolation, if/when we build it (the reversibility argument)

You can build isolation and then *choose* to share (a tenant publishes
into a commons — a shared lane, an explicit publish verb). You **cannot**
build a commons and then retrofit isolation — once instances have written
into a shared substrate with no per-principal boundary, the data is
commingled and there is no key to partition on after the fact. Isolation
is the strictly more general primitive; the commons is a policy layered on
top of it. Under uncertainty, isolation is the choice that keeps both
futures open.

**State objects are the stronger case than memories.** A state object is
the instance's working context — closest to "what it is mid-thought." If
instances share memory but not state, that is a commune. If they share
*state*, there is no distinct instance at all — just one mind with several
mouths. State-object privacy is therefore not a feature of multi-instance;
it is the precondition that makes "an instance" a coherent unit, and the
precondition for #13's "attribute to a distinct instance" to mean anything.

## The decision: defer, and prove the single instance first

**We do NOT build multi-instance isolation now.** The deliberate sequence:
*see whether a single long-horizon instance is useful at all before
building a community.* YAGNI, and we will have better insight after living
with one instance than we have guessing now.

### The discipline that makes the deferral safe

Deferring the *build* is correct. Foreclosing the *future* is not. The
failure mode is premature collapse by accretion: a useful single instance
quietly accreting code and data that assume exactly one principal, until
"build the community" discovers the single-instance success has been
foreclosing the multi-instance future all along.

The single-instance phase is therefore allowed to **not build** the
boundary, but is **not allowed** to:

- write data whose shape assumes the boundary will never exist (keep
  principal-shaped fields present-but-unenforced, e.g. `author_instance_id`,
  `authorship_verified`);
- optimize any query on the assumption that all records share one author;
- silently mark as verified/owned anything whose verification/ownership is
  not actually established.

**These three prohibitions MUST be red-bar tests, not prose.** This is the
document's own diagnosis turned on itself: the failure mode named above is
*premature collapse by accretion* — negative requirements eroding at the
margin between sessions because no local optimizer defends them. A list of
"not allowed to"s is exactly the defense we have learned does not hold. A
future instance will optimize a query on the single-author assumption
because it is faster, and nobody will notice until the rebuild finds the
corpus commingled. So each prohibition is owed a structural guard:

- a test that fails if a record is written without the principal-shaped
  fields present;
- a test that fails if a query is added that is correct only under the
  single-author assumption (harder to express mechanically — at minimum a
  red bar asserting the corpus contains, or could contain, more than one
  distinct `author_instance_id`, so no query may assume singularity);
- a test that fails if `authorship_verified` is ever `True` without a real
  identity source (today: always — nothing may flip it; this is the same
  guard as the Llika-slice Hole-1 red bar).

Until those tests exist, the prohibitions are prose, and this document is
making the mistake it diagnoses. Filing them is a precondition of the
discipline being real, not an optional follow-up.

### What the rebuild can and cannot recover

The eventual rebuild is a rebuild *of mechanism* (enforcement, topology),
not *of history* — **but only to the extent the unverified principal
labels turn out to have been accurate.** This is a contingency, not a
guarantee, and the reassuring version of this sentence is dangerous:

Isolation requires database-per-instance topology (above). At rebuild, a
corpus written single-principal into one database must be partitioned
across N databases *by principal* — and the only partition key is
`author_instance_id`, which is **self-asserted and unverified**. You
cannot reliably partition a commingled corpus on a field whose values were
never verified. If two instances ever wrote records both claiming
`author_instance_id: A` (which the current substrate *permits* — that is
the whole finding), the rebuild has no way to recover which were really
A's. The corpus does not independently survive; it survives only as well
as the labels happen to be honest.

What makes the labels honest *in practice* is precisely that there is
**one instance**: a sole author's self-assertions cannot collide with
another's. But that is a property of the single-instance invariant
*holding* — including against bugs, not just against a second instance —
**right up until the rebuild**, not a property the stored data carries.
The moment a second writer exists (or a bug forges a label), every prior
label becomes retroactively unreliable. So the cost of deferral grows with
time: the longer the single instance runs, the more the corpus's
survivability rests on the invariant never having been violated even once.
Mark the holes and keep the fields honest — that maximizes the labels'
trustworthiness — but the survival of history is *contingent on the
single-instance invariant*, and the substrate does not guarantee it.

## Status

- **Standing:** deferred, pending evidence that a single long-horizon
  instance earns its keep. Not scheduled. Not foreclosed.
- **Tracked:** yanantin#13 (whose title — "cannot attribute memory to a
  distinct instance" — undersells this; it is the *attribution* symptom of
  a *single-principal-substrate* cause). This doc is the fuller stance #13
  should point at.
- **Memory:** `pukara/memory/project_multitenancy_absent_verified.md`.
- **Open platform constraint:** ArangoDB sub-database access controls.
  Revisit if the platform gains them, or if the rebuild adopts
  database-per-instance / application-layer partitioning.
- **Accretion guards — INSTALLED as dormant red bars, not left as prose.**
  Rather than *specify* the guards and hope a future hand authors them
  (which decays: a spec is still prose, and an undischarged "MUST" teaches
  the next instance the MUST was optional), two of the three are written as
  executable `xfail(strict=True)` tests, now **placed and running** at
  `yanantin/tests/red_bar/test_single_principal_accretion.py` (yanantin
  commit `da34519`). Verified live: 1 passed, 1 xfailed.
  - Guard 1 (principal-shaped fields stay present) passes today; red if
    removed.
  - Guard 2 (`authorship_verified` exists and defaults False) xfails today
    because the field is unbuilt; `strict=True` flips the suite red the day
    it is added, forcing the guard to be made real. The wake-up is
    mechanical — nothing relies on a future instance remembering.
  - Guard 3 (no query optimized on the single-author assumption) is NOT a
    test — it is not a property a test can assert generally (a single-author
    query looks correct until a second author exists). It stays a
    **structural-review obligation** here, deliberately, not faked into a
    test that cannot check it.
  - The guards were authored by the Pukara hand and **placed by a yanantin
    hand** — the cross-fortress write (Pukara → yanantin/tests/) was
    declined on principle (author there, place here), because the
    benign-looking crossing is the shape every erosion takes.

## Provenance of this document

Drafted by the Pukara instance (sent to do the Llika slice; wrote the
standing decision that reframes it). Two amendments above — the
red-bar-not-prose requirement, and the contingent-corpus-survival
correction — came from a yanantin adversarial review that caught the
document making, one level up, the exact prose-erosion mistake it
diagnoses. The corrections are load-bearing and are folded in. Recorded
so a future reader knows this was two hands, not one, and which parts each
contributed.
