from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def require_keys(config: Dict[str, Any], keys: Iterable[str], config_path: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing required config key(s) in {config_path}: {joined}")


def require_existing_file(path: str, field_name: str) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"Config field '{field_name}' must be a file path, got {path!r}")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config field '{field_name}' must point to an existing file: {resolved}")
    return resolved


def require_existing_dir(path: str, field_name: str) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"Config field '{field_name}' must be a directory path, got {path!r}")
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Config field '{field_name}' must point to an existing directory: {resolved}")
    return resolved


def collect_input_files(input_dir: str, glob_pattern: str = "*.png") -> List[str]:
    root = require_existing_dir(input_dir, "input_dir")
    paths = sorted(str(path) for path in root.glob(glob_pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No input files matching '{glob_pattern}' found under {root}")
    return paths
