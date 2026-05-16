# Pukara Security Gateway Installation Plan

Date: 2026-05-16
Status: Draft for review
Related plan: Yanantin cross-project package boundaries, 2026-05-16

## Purpose

This plan describes how Pukara should become an installable security gateway that can stand on its own while remaining useful to Yanantin and compatible with Willay.

The guiding boundary is simple: Pukara is not the memory system and is not the attestation system. It is the enforcement layer that reduces the risk surface area of third-party storage by keeping semantic schema meaning away from the storage provider.

## Product Statement

Pukara is a storage-confidentiality gateway for systems that need to use external databases without giving those providers a readable semantic map of the data.

In the Yanantin ecosystem, Pukara protects memory infrastructure by separating the local mapping of semantic labels from the remote representation of collections and fields.

## Ayni For New Operators

Pukara's documentation should welcome careful operators by being explicit about
both protection and risk. A new user should not need to infer the security model
from code.

The docs and tools should:

- provide a local run path with visible success;
- name the secret materials that must be protected;
- make unsafe development settings unmistakable;
- state the "does not protect" list before deployment;
- keep Yanantin integration useful without making Pukara feel private to
  Yanantin.

## Current State

Pukara currently provides a FastAPI gateway with Apacheta-oriented routes, authentication, configuration, and tests. Recent schema-map work gives it a clearer security story: semantic names can be mapped to deterministic opaque identifiers before storage.

The repository is now in a good place to define installation and operator-facing documentation. The main gap is not the existence of the idea, but a clear public path for someone else to install it, configure it, understand what it protects, and understand what it does not protect.

## Package Boundary

Pukara should be independently installable as its own package and service.

Recommended package shape:

- `pukara`: core gateway, configuration, schema mapping, and FastAPI app.
- `pukara[dev]`: tests, linting, and local development tools.
- Future `pukara[arango]` or equivalent only if storage-driver dependencies become optional.
- Future deployment artifact: container image, systemd unit, or both.

The package should avoid becoming a Yanantin plugin only. Yanantin can depend on Pukara as an optional storage-protection layer, but Pukara's public value is broader than Yanantin.

## Responsibilities

Pukara should own:

- API boundary enforcement.
- Authentication and service identity checks.
- Configuration loading and validation.
- Schema-map generation and lookup.
- Opaque storage naming for collections and fields.
- Audit logging of gateway operations.
- Operator documentation for safe installation.
- Backup and restore guidance for mapping data and mapping keys.
- Clear development-only transparent mode, if retained.

Pukara should not own:

- Yanantin memory semantics.
- Willay receipt semantics or ledger policy.
- Long-term institutional attestation.
- Application-level authorization decisions beyond gateway access rules.
- Content encryption unless that becomes an explicit later feature.

## Threat Model

The immediate adversary is a third-party database provider, database operator, or database dump reader who can inspect stored collection names, field names, and data shapes.

Pukara should protect:

- Semantic collection names.
- Semantic field names.
- Direct schema readability by the storage provider.
- Accidental leakage through provider dashboards, logs, exports, and support tooling.

Pukara does not currently protect:

- Plaintext content values.
- Traffic observed before TLS termination.
- A compromised gateway host.
- A compromised client device.
- A leaked mapping database or mapping key.
- Inference from value distributions, record counts, timing, or access patterns.

The mapping database and `PUKARA_STORAGE_KEY` are crown-jewel materials. Installation docs should treat them accordingly.

## Installation Story

The public installation path should let a new operator answer five questions quickly:

1. What does Pukara protect?
2. What does Pukara not protect?
3. What secret material must I keep safe?
4. How do I run it locally?
5. How do I put it in front of a database without accidentally disabling the protections?

Recommended documentation flow:

- README: short description, threat model summary, quick start, and next-step links.
- `docs/install.md`: local install, environment variables, service launch, health check.
- `docs/configuration.md`: config precedence, required variables, development defaults, production cautions.
- `docs/security.md`: threat model, mapping-key handling, backup/restore, transparent-mode warnings.
- `docs/integrations/yanantin.md`: how Yanantin should use Pukara.
- `docs/operators.md`: operational checklist for deployment and maintenance.

The first public story can remain Apacheta/Yanantin-oriented if that is the honest current surface. The docs should still frame Pukara as a general gateway so the later broadening does not require a conceptual rewrite.

## Configuration Work

Configuration should make unsafe states visible.

Needed decisions:

- Define whether environment variables, config files, or command-line flags have precedence.
- Provide a documented way to generate `PUKARA_STORAGE_KEY`.
- Fail clearly when production mode lacks required secrets.
- Make transparent mode explicitly development-only.
- Decide whether transparent mode should require loopback binding or an explicit unsafe flag.
- Document where mapping data lives and how it is backed up.
- Document how a lost mapping key affects recovery.

## Packaging Work

Package hardening should include:

- Review `pyproject.toml` metadata for PyPI readiness.
- Decide the minimum supported Python version.
- Pin or bound any Yanantin dependency if Pukara imports Yanantin public types.
- Keep test fixtures, local data, and operational secrets out of sdists and wheels.
- Add a console entry point if `python -m pukara` is not the intended operator command.
- Include docs that are needed for source distributions.

## Yanantin Integration

Yanantin should treat Pukara as an optional storage boundary, not as part of memory semantics.

The likely integration surface is a small client or backend adapter:

- Yanantin sends storage operations to Pukara.
- Pukara maps semantic schema names to opaque storage names.
- The database provider sees only the opaque representation.
- Yanantin retains the human-meaningful memory model.

This keeps the memory project teachable while allowing security-conscious operators to add a stronger boundary.

## Willay Integration

Willay should not depend on Pukara for attestation semantics.

Useful later integrations are still possible:

- Store Willay receipts behind Pukara.
- Attest Pukara configuration snapshots with Willay.
- Record Pukara mapping lifecycle events as attested operational events.

Those are integrations, not package-boundary requirements.

## Tests And Invariants

The installation work should preserve or add tests for these invariants:

- In opaque mode, semantic collection names do not appear in provider-facing storage names.
- In opaque mode, semantic field names do not appear in provider-facing stored records.
- The same semantic label maps deterministically under the same key.
- Different keys produce different opaque identifiers.
- Transparent mode cannot be mistaken for production mode.
- Gateway routes do not bypass schema mapping.
- Gateway clients cannot directly request privileged database operations.
- Mapping backup and restore can recover access to existing opaque data.

## First Implementation Pass

1. Update README with a short doorway for operators.
2. Add `docs/security.md` with the explicit threat model.
3. Add `docs/install.md` with a local quick start.
4. Add `docs/configuration.md` with config precedence and required secrets.
5. Add a key-generation helper or documented command.
6. Add focused tests for production-mode refusal and transparent-mode warnings.
7. Review packaging metadata and source distribution contents.

## Open Questions

- Should Pukara's first public installation story be explicitly Apacheta/Yanantin-specific, or should it present a generic gateway interface immediately?
- Should `PUKARA_STORAGE_KEY` be the only mapping secret, or should deployments support a separate key-management provider from the beginning?
- Should transparent mode exist in published packages, or only in tests and local development?
- Should Pukara eventually support content encryption, or should it remain narrowly focused on schema confidentiality?
- What is the minimum viable deployment target: `uv run`, console command, container, systemd unit, or hosted service?

## Review Points

The most important review points are:

- Whether the threat model is stated accurately.
- Whether the "does not protect" list is honest and complete.
- Whether Pukara should be positioned initially as Apacheta-specific or broadly installable.
- Whether content encryption belongs on the roadmap or should remain explicitly out of scope.
- Whether the installation path should prioritize local developers, self-hosting operators, or Yanantin users first.
