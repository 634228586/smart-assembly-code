from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from voice.qwen_recognizer import (
    RecognitionError,
    build_recognition_result,
    recognize_task_card_with_diagnostics,
)

from .config import endpoints, load_all
from .paths import SESSION_DIR
from .protocols import ProtocolError, validate_recognition_result
from .vision_client import RealVisionClient


class TaskCardModelTestWorker(QObject):
    """Capture one real task-card frame and test the formal model path without execution."""

    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all()
            configured = endpoints(configs["endpoints"])
            if "vision_service" not in configured:
                raise RuntimeError("vision_service端点尚未配置。")
            session_id = f"model-test-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            request_id = f"{session_id}-capture"
            session_dir = SESSION_DIR / session_id
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            )
            vision.health()
            frame = vision.capture_task_card(
                request_id=request_id, session_id=session_id, session_dir=session_dir,
            )
            base: dict[str, Any] = {
                "session_id": session_id,
                "request_id": request_id,
                "captured_at": frame.captured_at,
                "image_path": str(frame.image_path.resolve()),
            }
            started = time.perf_counter()
            try:
                diagnostic = recognize_task_card_with_diagnostics(frame.image_path)
                model_result = diagnostic["model_result"]
                formal = build_recognition_result(model_result, frame=frame)
                validated = validate_recognition_result(formal, session_dir=session_dir)
                self.finished.emit({
                    **base,
                    "success": True,
                    "recognition_success": validated.get("success") is True,
                    "model": diagnostic["model"],
                    "provider_request_id": diagnostic["provider_request_id"],
                    "elapsed_ms": diagnostic["elapsed_ms"],
                    "raw_response": diagnostic["raw_response"],
                    "model_result": model_result,
                    "formal_result": formal,
                    "validated_result": validated,
                    "validation_message": "已通过正式任务卡协议校验；测试结果未进入比赛会话。",
                })
            except (RecognitionError, ProtocolError) as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                self.finished.emit({
                    **base,
                    "success": False,
                    "recognition_success": False,
                    "model": getattr(exc, "model", None),
                    "provider_request_id": None,
                    "elapsed_ms": elapsed_ms,
                    "raw_response": getattr(exc, "raw_response", None),
                    "model_result": None,
                    "formal_result": None,
                    "validated_result": None,
                    "validation_message": f"未通过正式识别/协议校验：{type(exc).__name__}: {exc}",
                })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
