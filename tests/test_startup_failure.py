"""Startup fail-stop boundary tests for the Pukara gateway lifespan.

The change under test lives in ``pukara.app.lifespan``: backend
construction is wrapped in ``try/except ConnectionError`` so the
discriminated upstream failures (auth / unreachable / not-provisioned)
get an honest, visible diagnosis at the boundary before fail-stop
re-raises. The except clause is *specifically* ``ConnectionError`` —
not ``Exception`` — so a bug in our own startup propagates raw rather
than being mislabeled "Backend startup failed".

These tests stub ``pukara.app.ArangoDBBackend`` (the name bound into
``app.py``'s module namespace) so no live ArangoDB is needed. The
lifespan fires on ``TestClient`` context-manager entry, so entering the
context is what surfaces a startup failure.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from yanantin.apacheta.interface.errors import (
    BackendAuthError,
    BackendUnreachableError,
    DatabaseNotProvisionedError,
)

import pukara.app as app_module
from pukara.app import create_app
from pukara.config import PukaraConfig


def _dev_config(**overrides) -> PukaraConfig:
    """Minimal valid config: empty api_key (dev mode), empty storage_key
    (transparent obfuscation), so the lifespan reaches backend
    construction without auth or UUID parsing getting in the way."""
    base: dict[str, object] = dict(
        arango_host="http://unreachable.invalid:8529",
        arango_db="apacheta",
        arango_user="apacheta_app",
        arango_password="",
        api_key="",
        storage_key="",
    )
    base.update(overrides)
    return PukaraConfig(**base)  # type: ignore[arg-type]


def _raising_backend(exc: BaseException):
    """Return a stub class whose __init__ raises ``exc``, matching the
    keyword signature ArangoDBBackend is constructed with in lifespan."""

    class _StubBackend:
        def __init__(self, *, host, db_name, username, password, obfuscator):
            raise exc

        def close(self):  # pragma: no cover - never reached on failure
            pass

    return _StubBackend


# ── A. Discriminated ConnectionError subclasses ───────────────────────
#
# Each surfaces through the boundary (fail-stop holds) AND its own class
# name — not a generic label — is what the boundary log line carries.

_SUBCLASSES = [
    BackendAuthError,
    BackendUnreachableError,
    DatabaseNotProvisionedError,
]


@pytest.mark.parametrize(
    "exc_cls", _SUBCLASSES, ids=[c.__name__ for c in _SUBCLASSES]
)
def test_connection_error_subclass_propagates_and_is_logged(
    monkeypatch, caplog, exc_cls
):
    exc = exc_cls("upstream remediation message")
    monkeypatch.setattr(app_module, "ArangoDBBackend", _raising_backend(exc))

    app = create_app(_dev_config())

    with caplog.at_level(logging.ERROR, logger="pukara"):
        # Requirement 1: fail-stop — entering the context propagates the
        # exact exception instance the backend raised.
        with pytest.raises(exc_cls) as excinfo:
            with TestClient(app):
                pass
    assert excinfo.value is exc

    # Requirement 2: the boundary log line is emitted at ERROR on logger
    # "pukara" and carries the exception CLASS NAME.
    boundary_records = [
        r
        for r in caplog.records
        if r.name == "pukara"
        and r.levelno == logging.ERROR
        and "Backend startup failed" in r.getMessage()
    ]
    assert len(boundary_records) == 1
    message = boundary_records[0].getMessage()
    assert exc_cls.__name__ in message


def test_bare_connection_error_also_surfaces_with_its_name(monkeypatch, caplog):
    """The fallthrough: a plain ConnectionError (unrecognized failure)
    still hits the except clause, gets logged with its class name, and
    re-raises."""
    exc = ConnectionError("unrecognized failure")
    monkeypatch.setattr(app_module, "ArangoDBBackend", _raising_backend(exc))

    app = create_app(_dev_config())

    with caplog.at_level(logging.ERROR, logger="pukara"):
        with pytest.raises(ConnectionError) as excinfo:
            with TestClient(app):
                pass
    assert excinfo.value is exc

    boundary_records = [
        r
        for r in caplog.records
        if r.name == "pukara" and "Backend startup failed" in r.getMessage()
    ]
    assert len(boundary_records) == 1
    assert "ConnectionError" in boundary_records[0].getMessage()


# ── B. Non-ConnectionError must NOT be caught or relabeled ─────────────
#
# The except clause is ConnectionError, not Exception. A backend that
# fails some other way (e.g. a privilege/provisioning error surfaced as
# RuntimeError, or a programming ValueError) must propagate RAW: the
# original type, and no "Backend startup failed" log line claiming
# Pukara diagnosed it.

_NON_CONNECTION = [
    RuntimeError("privilege error during _ensure_collections"),
    ValueError("backend construction blew up"),
    KeyError("config drift"),
]


@pytest.mark.parametrize(
    "exc", _NON_CONNECTION, ids=[type(e).__name__ for e in _NON_CONNECTION]
)
def test_non_connection_error_propagates_unlabeled(monkeypatch, caplog, exc):
    monkeypatch.setattr(app_module, "ArangoDBBackend", _raising_backend(exc))

    app = create_app(_dev_config())

    with caplog.at_level(logging.ERROR, logger="pukara"):
        with pytest.raises(type(exc)) as excinfo:
            with TestClient(app):
                pass
    # Same instance, unchanged — not wrapped, not relabeled.
    assert excinfo.value is exc

    # No boundary log line — Pukara did NOT claim to diagnose this.
    assert not [
        r for r in caplog.records if "Backend startup failed" in r.getMessage()
    ]


# ── C. Failure before backend construction propagates unlabeled ────────
#
# A malformed storage_key is parsed via UUID(...) at the very top of the
# lifespan, before ArangoDBBackend is ever called. That ValueError must
# also propagate raw, with no "Backend startup failed" label. We assert
# the backend is never even constructed.


def test_malformed_storage_key_propagates_before_backend(monkeypatch, caplog):
    constructed = {"called": False}

    class _TrippedBackend:
        def __init__(self, *, host, db_name, username, password, obfuscator):
            constructed["called"] = True

        def close(self):  # pragma: no cover
            pass

    monkeypatch.setattr(app_module, "ArangoDBBackend", _TrippedBackend)

    app = create_app(_dev_config(storage_key="not-a-uuid"))

    with caplog.at_level(logging.ERROR, logger="pukara"):
        with pytest.raises(ValueError):
            with TestClient(app):
                pass

    # Backend was never reached, so the boundary except never fired.
    assert constructed["called"] is False
    assert not [
        r for r in caplog.records if "Backend startup failed" in r.getMessage()
    ]
