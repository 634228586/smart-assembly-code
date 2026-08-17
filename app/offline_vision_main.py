from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QSG_RHI_BACKEND", "software")


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
        from .offline_vision_tool import OfflineVisionWindow, run_offline_vision_tool
        if os.environ.get("OFFLINE_VISION_STARTUP_CHECK_ONLY") == "1":
            app = QApplication.instance() or QApplication([])
            window = OfflineVisionWindow(); window.close()
            return 0
        return run_offline_vision_tool()
    except Exception as exc:
        print(f"离线视觉工具启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
