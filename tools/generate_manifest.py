from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".runtime", ".pytest_cache", "__pycache__", "logs", "sessions"}
EXCLUDED_FILES = {"data/sessions.zip"}


def main() -> None:
    lines: list[str] = []
    for path in sorted(ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == OUTPUT or relative.as_posix() in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest} *{relative.as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"已生成 {OUTPUT.name}：{len(lines)} 个文件")


if __name__ == "__main__":
    main()
