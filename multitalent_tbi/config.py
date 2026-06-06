from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(path: str | Path) -> SimpleNamespace:
    return _to_namespace(load_yaml(path))


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
