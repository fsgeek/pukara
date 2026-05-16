# Schema Extras and the Registration Question

*Captured 2026-05-16 — state of the conversation, not a finalized design.*

A drift-finding session that started with a reviewer's note about a
stale Pukara test (`test_post_with_extra_fields_rejected`) and walked
backwards into a real architectural seam in `SchemaMap`. Three threads
came out. Two are settled enough to act on; one is deferred until a
tiered-trust auth model exists.

## What Surfaced

### 1. Test asserts behavior that no longer exists

`tests/test_gateway_independent.py:1343-1347` asserts that posting a
record with an extra field returns 422. It currently returns 201 and
the test fails. The behavior was deliberately removed upstream: tiksi's
`ApachetaBaseModel` was flipped from `extra="forbid"` to
`extra="allow"` in yanantin commit `2661099d` ("Open tensor schema") —
the hamut'ay-taste-open insight that says restricting schema extensions
in a system that needs dynamic extensibility is anti-pattern. Tiksi
inherited the open form when it was extracted from yanantin.

The test is stale. Deletion is correct in principle, but the right
*replacement* depends on what the gateway is supposed to do about
extras, which is the next thread.

### 2. SchemaMap field/document policy mismatch — the real seam

`src/pukara/schema_map.py` has two methods that disagree about
unknown fields:

- `field_name()` at line 145 ("Unknown field names are computed and
  cached on the fly") — *dynamic registration on demand.*
- `_obfuscate_recursive()` at line 215 ("Map known keys to opaque
  form. Unknown keys pass through.") — *consults the cache directly;
  does not register.*

The pre-registration step at startup (`_collect_model_field_names`)
walks every known Pydantic model and registers their field names. With
the old `extra="forbid"` policy, those were the only fields that could
ever appear in a document. With the new `extra="allow"` policy, extras
ride along through pydantic validation, through `_obfuscate_recursive`
(unchanged), and into ArangoDB *with cleartext field names* alongside
the obfuscated real ones. That breaks SchemaMap's threat model — a
compromised database reveals exactly the labels the obfuscation was
meant to hide.

The current half-implementation is the worst of both worlds: it has
the friction of pre-declaration (only pre-registered fields get
obfuscated) without the safety of pre-declaration (unknowns aren't
rejected, they leak through).

### 3. Wire-level error detail — deferred

`src/pukara/app.py:83-101` returns `{"detail": str(exc)}` for every
custom exception handler, and FastAPI's default 422 handler returns
the full pydantic validation structure with field names. If Pukara
ever rejected extras with the default 422, the response would
enumerate the schema for the caller.

**This is not a fix for today.** The threat model Pukara was built
under puts the database on the untrusted side; the authenticated
caller is trusted. The wire-level error surface only matters to an
attacker who is itself an authenticated caller — which requires a
tiered-trust auth model that doesn't yet exist (current auth is
binary: valid key = full trust, empty = dev). Deferred until tiers.

## What's Settled

**The extension path is explicit upstream registration, not silent
strip-at-boundary.** The choices on the table were:

- *(a)* Strip unknowns at the Pukara boundary, silently drop. Audit-log
  for debugging.
- *(b)* Have SchemaMap obfuscate unknowns on the fly (extend the
  dynamic-registration intent of `field_name()` to documents too).
- *(c)* Declare the loss — document that schema obfuscation does not
  cover smuggled extras.

The silent strip in *(a)* is gateway-takes-without-giving: the
producer loses data, gets a 201, has no debugging path. That's an
ayni violation; it makes Pukara opaque in the way "security as
performative trust" tends to. The reject-as-security-violation
variant is harsh and re-imposes the friction that `extra="allow"`
was meant to remove.

The principled path is *explicit registration as a positive,
substrate-level action.* The substrate consents to a field set; the
gateway enforces what was opened. Hamut'ay is preserved (the schema
can still grow), and ser seguro rather than estar seguro — the safety
is structural (the registration set is what it is), not performative
(Pukara claims to keep you safe and you must trust the claim).

## What This Probably Looks Like

Sketch, not commitment:

- A registration endpoint or substrate-level call that adds a field
  name to the SchemaMap field cache. Returns the opaque mapping. The
  registration is the public act of opening the schema; it is auditable
  because it is a call, not a side-effect of a write.
- On write, `_obfuscate_recursive` enforces what was registered:
  unknown top-level keys cause the write to fail with a clear error
  (in audit logs and to the registered caller — the threat model
  doesn't penalize informative errors here because the database isn't
  reading them).
- Known dict-typed fields like `identity_data` continue to pass
  arbitrary keys through unchanged. Hamut'ay-via-`metadata: dict`
  still works for value extensibility; the registration step is for
  *new modeled fields,* not for free-form data inside known fields.
- Audit log records both the registration calls and any rejected
  writes. Operators see schema growth as a sequence of explicit acts.

Open questions:

- Who is authorized to register? With binary auth today, any
  API-key holder can. With future tiered auth, registration might be
  a higher tier than write.
- Is registration reversible? (Probably not — once a field is in the
  cache, removing it would invalidate stored documents that use the
  opaque mapping. Renaming/aliasing maybe.)
- Does registration cross deployments? (The UUID namespace is
  per-deployment, so the same semantic name yields different opaque
  names in different deployments. Registration is local.)

## Immediate Actions

Three things can move now without further design:

1. **Delete the stale test.** `tests/test_gateway_independent.py:1343-1347`.
   Builder/tester separation rule means this should be a separate
   commit from any src/ changes, ideally by a different role than
   the one editing src/.
2. **Update `MEMORY.md`** to reflect this conversation. Done in
   parallel with this plan.
3. **Hold on implementing registration** until the design above is
   approved and the open questions are answered.

## Tests That Will Become Stale When Registration Lands

When the registration design above is implemented, the following
tests pin the *current* (pass-through) behavior and will need to be
flipped or deleted as part of that work — don't lose them:

- `tests/test_schema_map.py::test_unknown_keys_pass_through_in_obfuscate`
- `tests/test_schema_map.py::test_unknown_keys_pass_through_in_deobfuscate`

Both explicitly assert that unknown top-level keys flow through
`SchemaMap` unchanged. The registration design *removes* that
behavior — known-registered fields are obfuscated, unknown ones
either fail the write or trigger a registration call, depending on
which variant we pick. Either way these assertions become wrong.

## Related Memory

- `memory/feedback_threat_model_default.md` — adversary framing
- `memory/project_schema_registration_direction.md` — design direction
- `memory/feedback_hedging_after_initiative.md` — process note
