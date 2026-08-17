from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG_DIR = PACKAGE_ROOT / "config" / "real"
REAL_CALIBRATION_DIR = PACKAGE_ROOT / "calibration" / "real"
DATA_DIR = PACKAGE_ROOT / "data"
LOG_DIR = PACKAGE_ROOT / "logs"
SESSION_DIR = DATA_DIR / "sessions"
EVIDENCE_DIR = LOG_DIR / "evidence"


def portable_project_path(path: str | Path) -> str:
    """Store project-owned files without binding records to one computer."""

    value = Path(path)
    try:
        return value.resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(value)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve current relative paths and legacy absolute project paths."""

    value = Path(path)
    if not value.is_absolute():
        return (PACKAGE_ROOT / value).resolve()
    if value.is_file():
        return value.resolve()

    parts = value.parts
    lowered = tuple(part.casefold() for part in parts)
    for marker in (("data", "sessions"), ("logs", "evidence"), ("calibration", "real")):
        for index in range(len(lowered) - len(marker) + 1):
            if lowered[index:index + len(marker)] == marker:
                return (PACKAGE_ROOT.joinpath(*parts[index:])).resolve()
    return value


def ensure_runtime_directories() -> None:
    """只创建输出目录，不创建或填充任何硬件配置。"""

    for path in (SESSION_DIR, EVIDENCE_DIR, LOG_DIR / "controller", LOG_DIR / "vision", LOG_DIR / "voice"):
        path.mkdir(parents=True, exist_ok=True)
