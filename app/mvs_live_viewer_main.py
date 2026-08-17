from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QSG_RHI_BACKEND", "software")


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
        from .mvs_live_viewer import MvsLiveViewerWindow, run_mvs_live_viewer

        if os.environ.get("MVS_LIVE_VIEWER_STARTUP_CHECK_ONLY") == "1":
            app = QApplication.instance() or QApplication([])
            window = MvsLiveViewerWindow(auto_start=False)
            window.close()
            return 0
        return run_mvs_live_viewer()
    except Exception as exc:
        print(f"MVS实时画面启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
