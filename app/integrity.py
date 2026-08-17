from __future__ import annotations

import hashlib
from pathlib import Path

from .config import canonical_hash
from .paths import PACKAGE_ROOT, REAL_CALIBRATION_DIR, REAL_CONFIG_DIR


def _relative_tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix().lower()):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def current_config_hash() -> str:
    return canonical_hash(path for path in REAL_CONFIG_DIR.glob("*.json") if path.name != "baseline.json")


def current_calibration_hash() -> str:
    return _relative_tree_hash([path for path in REAL_CALIBRATION_DIR.rglob("*.json") if path.is_file()])


def current_code_hash() -> str:
    files: list[Path] = []
    for directory in (PACKAGE_ROOT / "app", PACKAGE_ROOT / "vision", PACKAGE_ROOT / "voice"):
        files.extend(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)
    for name in ("requirements-lock.txt", "start_competition.ps1", "启动正式比赛.ps1", "启动正式比赛.cmd", "VERSION.json"):
        path = PACKAGE_ROOT / name
        if path.is_file():
            files.append(path)
    return _relative_tree_hash(files)


def current_runtime_fingerprint() -> str:
    digest = hashlib.sha256()
    for value in (current_code_hash(), current_config_hash(), current_calibration_hash()):
        digest.update(value.encode("ascii")); digest.update(b"\0")
    return digest.hexdigest()
