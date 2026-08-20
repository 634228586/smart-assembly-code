from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from vision.mvs_camera import MvsCamera

from .config import load_json
from .paths import REAL_CONFIG_DIR


class LiveCaptureWorker(QObject):
    frame_ready = Signal(object, int, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        profile: dict[str, Any],
        camera_factory: Callable[[], Any] = MvsCamera,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.camera_factory = camera_factory
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    @Slot()
    def run(self) -> None:
        camera = None
        try:
            camera = self.camera_factory()
            camera.open_first_available()
            camera.start_preview(self.profile)
            previous = time.perf_counter()
            while not self._stop_event.is_set():
                capture = camera.read_preview_frame(timeout_ms=500)
                current = time.perf_counter()
                fps = 1.0 / max(current - previous, 1e-6)
                previous = current
                self.frame_ready.emit(capture.image_bgr, capture.frame_number, fps)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self.failed.emit(f"关闭相机失败：{exc}")
            self.finished.emit()


class MvsLiveViewerWindow(QMainWindow):
    def __init__(
        self,
        *,
        auto_start: bool = True,
        camera_config_path: Path | None = None,
        camera_factory: Callable[[], Any] = MvsCamera,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MVS 相机实时画面")
        self.resize(1100, 760)
        self.camera_factory = camera_factory
        self.camera_config_path = camera_config_path or (REAL_CONFIG_DIR / "camera.json")
        self._thread: QThread | None = None
        self._worker: LiveCaptureWorker | None = None
        self._last_image: np.ndarray | None = None
        self._last_frame_number = 0
        self._last_fps = 0.0

        self.image_label = QLabel("正在连接 MVS 相机……" if auto_start else "MVS 实时预览")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet("QLabel { background: #171717; color: #dddddd; font-size: 18px; }")
        self.status_label = QLabel("仅显示画面；不保存、不识别、不连接机械臂")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)
        self.setCentralWidget(container)

        if auto_start:
            self.start_preview()

    def start_preview(self) -> None:
        if self._thread is not None:
            return
        config = load_json(self.camera_config_path)
        profiles = config.get("profiles")
        if not isinstance(profiles, dict) or not isinstance(profiles.get("blocks"), dict):
            raise ValueError("camera.json缺少 blocks参数。")
        thread = QThread(self)
        worker = LiveCaptureWorker(
            profile=dict(profiles["blocks"]),
            camera_factory=self.camera_factory,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.frame_ready.connect(self._show_frame)
        worker.failed.connect(self._show_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._preview_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object, int, float)
    def _show_frame(self, image_bgr: np.ndarray, frame_number: int, fps: float) -> None:
        image = np.ascontiguousarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            self._show_error("相机返回了无效彩色帧。")
            return
        self._last_image = image.copy()
        self._last_frame_number = int(frame_number)
        self._last_fps = float(fps)
        height, width = image.shape[:2]
        rgb = image[:, :, ::-1].copy()
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.status_label.setText(f"实时画面｜帧号 {frame_number}｜{fps:.1f} FPS｜blocks参数")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self.image_label.clear()
        self.image_label.setText("无法显示相机画面")
        self.status_label.setText(message)

    @Slot()
    def _preview_finished(self) -> None:
        thread = self._thread
        self._worker = None
        self._thread = None
        if thread is not None:
            thread.deleteLater()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._last_image is not None:
            self._show_frame(self._last_image, self._last_frame_number, self._last_fps)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2500)
        event.accept()


def run_mvs_live_viewer() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MvsLiveViewerWindow()
    window.show()
    return app.exec()
