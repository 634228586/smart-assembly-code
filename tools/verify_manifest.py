from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".runtime", ".pytest_cache", "__pycache__", "logs", "sessions"}
EXCLUDED_FILES = {"data/sessions.zip"}


def main() -> int:
    failures: list[str] = []
    listed: set[str] = set()
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(" *", 1)
        except ValueError:
            failures.append(f"line {line_number}: invalid format")
            continue
        path = (ROOT / relative).resolve()
        listed.add(Path(relative).as_posix())
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            failures.append(f"line {line_number}: path escapes package")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"changed: {relative}")
    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FILES
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
    }
    for relative in sorted(actual_files - listed):
        failures.append(f"unlisted: {relative}")
    for relative in sorted(listed - actual_files):
        failures.append(f"stale manifest entry: {relative}")
    if failures:
        print("\n".join(failures))
        return 1
    print("MANIFEST verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
