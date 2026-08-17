from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.mvs_live_viewer import MvsLiveViewerWindow


class MvsLiveViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_can_start_without_opening_hardware_and_display_frame(self) -> None:
        window = MvsLiveViewerWindow(auto_start=False)
        try:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, 1] = 180
            window._show_frame(frame, 17, 24.5)
            self.app.processEvents()
            self.assertIsNotNone(window.image_label.pixmap())
            self.assertIn("帧号 17", window.status_label.text())
            self.assertIn("24.5 FPS", window.status_label.text())
            self.assertIsNone(window._thread)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
