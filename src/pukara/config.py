"""Configuration for Pukara gateway.

Reads from config/pukara.ini if present, environment variables override.
Credentials never checked into version control.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "pukara.ini"


@dataclass(frozen=True)
class PukaraConfig:
    """Gateway configuration. Immutable once loaded."""

    arango_host: str = "http://192.168.111.125:8529"
    arango_db: str = "apacheta"
    arango_user: str = "root"
    arango_password: str = ""
    api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 8000


def load_config(config_path: Path | None = None) -> PukaraConfig:
    """Load config from INI file, then override with environment variables."""
    path = config_path or _CONFIG_FILE
    kwargs: dict = {}

    if path.exists():
        parser = configparser.ConfigParser()
        parser.read(path)
        if parser.has_section("arango"):
            for ini_key, config_key in [
                ("host", "arango_host"),
                ("database", "arango_db"),
                ("username", "arango_user"),
                ("password", "arango_password"),
            ]:
                val = parser.get("arango", ini_key, fallback=None)
                if val is not None:
                    kwargs[config_key] = val
        if parser.has_section("gateway"):
            for ini_key, config_key in [
                ("api_key", "api_key"),
                ("host", "host"),
            ]:
                val = parser.get("gateway", ini_key, fallback=None)
                if val is not None:
                    kwargs[config_key] = val
            port_str = parser.get("gateway", "port", fallback=None)
            if port_str is not None:
                kwargs["port"] = int(port_str)

    env_map = {
        "PUKARA_ARANGO_HOST": "arango_host",
        "PUKARA_ARANGO_DB": "arango_db",
        "PUKARA_ARANGO_USER": "arango_user",
        "PUKARA_ARANGO_PASSWORD": "arango_password",
        "PUKARA_API_KEY": "api_key",
        "PUKARA_HOST": "host",
        "PUKARA_PORT": "port",
    }
    for env_key, config_key in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            if config_key == "port":
                kwargs[config_key] = int(val)
            else:
                kwargs[config_key] = val

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return PukaraConfig(**kwargs)
