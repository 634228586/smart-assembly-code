from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QSG_RHI_BACKEND", "software")

from .paths import ensure_runtime_directories


def main() -> int:
    ensure_runtime_directories()
    try:
        from .ui import run_app
        return run_app()
    except Exception as exc:
        print(f"正式比赛程序启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
