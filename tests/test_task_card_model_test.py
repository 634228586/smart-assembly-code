from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from vision.contracts import CapturedFrame
from voice.qwen_recognizer import (
    MODEL_IMAGE_MAX_BYTES,
    MODEL_IMAGE_MAX_EDGE_PX,
    RecognitionError,
    _prepare_model_image,
    recognize_task_card_with_diagnostics,
)

from app.task_card_model_test_worker import TaskCardModelTestWorker


class TaskCardModelDiagnosticTest(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, *, width: int = 80, height: int = 60) -> bytes:
        image = np.full((height, width, 3), (40, 120, 220), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError("test image encoding failed")
        payload = encoded.tobytes()
        path.write_bytes(payload)
        return payload

    @staticmethod
    def _response(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            id="provider-request-1",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )

    def test_diagnostic_uses_formal_model_call_and_preserves_raw_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "task.png"; self._write_image(image)
            raw = '{"success":true,"task_type":"task_1","confidence":0.98,"scene_description":"测试场景"}'
            with (
                patch.dict("os.environ", {
                    "DASHSCOPE_API_KEY": "secret-test-key",
                    "DASHSCOPE_BASE_URL": "https://example.invalid/v1",
                    "DASHSCOPE_MODEL": "qwen-test-model",
                }, clear=False),
                patch("openai.OpenAI") as client_factory,
            ):
                client_factory.return_value.chat.completions.create.return_value = self._response(raw)
                result = recognize_task_card_with_diagnostics(image)
            self.assertEqual(result["raw_response"], raw)
            self.assertEqual(result["model_result"]["task_type"], "task_1")
            self.assertEqual(result["model"], "qwen-test-model")
            self.assertEqual(result["provider_request_id"], "provider-request-1")
            self.assertNotIn("secret-test-key", str(result))
            call = client_factory.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(call["temperature"], 0)
            self.assertEqual(call["model"], "qwen-test-model")
            image_url = call["messages"][1]["content"][1]["image_url"]["url"]
            self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
            self.assertTrue(image.with_name("task_qwen.jpg").is_file())

    def test_model_image_is_resized_bounded_and_original_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / "task.png"
            original_payload = self._write_image(original, width=3072, height=2048)
            prepared = _prepare_model_image(original)
            converted = cv2.imdecode(
                np.frombuffer(prepared.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(converted)
            self.assertLessEqual(max(converted.shape[:2]), MODEL_IMAGE_MAX_EDGE_PX)
            self.assertLessEqual(prepared.stat().st_size, MODEL_IMAGE_MAX_BYTES)
            self.assertEqual(original.read_bytes(), original_payload)
            self.assertEqual(prepared.name, "task_qwen.jpg")

    def test_invalid_json_keeps_raw_model_text_for_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "task.png"; self._write_image(image)
            with (
                patch.dict("os.environ", {
                    "DASHSCOPE_API_KEY": "secret-test-key",
                    "DASHSCOPE_BASE_URL": "https://example.invalid/v1",
                    "DASHSCOPE_MODEL": "qwen-test-model",
                }, clear=False),
                patch("openai.OpenAI") as client_factory,
            ):
                client_factory.return_value.chat.completions.create.return_value = self._response("not-json")
                with self.assertRaises(RecognitionError) as raised:
                    recognize_task_card_with_diagnostics(image)
            self.assertEqual(raised.exception.raw_response, "not-json")

    def test_worker_validates_formal_result_without_starting_competition(self) -> None:
        class FakeVisionClient:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            @staticmethod
            def health() -> dict:
                return {"status": "ready"}

            @staticmethod
            def capture_task_card(*, request_id: str, session_id: str, session_dir: Path) -> CapturedFrame:
                image = session_dir / "task_card" / "fresh.png"
                image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(b"fresh")
                return CapturedFrame(
                    request_id, "CAM-TEST", "task_card", image,
                    datetime.now(timezone.utc).isoformat(), True,
                )

        model_result = {
            "success": True, "task_type": "task_1", "confidence": 0.99,
            "scene_description": "模型测试场景",
        }
        with tempfile.TemporaryDirectory() as temp:
            worker = TaskCardModelTestWorker(); results: list[dict] = []; failures: list[str] = []
            worker.finished.connect(results.append); worker.failed.connect(failures.append)
            with (
                patch("app.task_card_model_test_worker.SESSION_DIR", Path(temp)),
                patch("app.task_card_model_test_worker.load_all", return_value={
                    "endpoints": {},
                    "camera": {"serial_number": "CAM-TEST", "fresh_frame_max_age_ms": 5000},
                    "robot": {"active_tcp": {"name": "TCP-TEST"}},
                }),
                patch("app.task_card_model_test_worker.endpoints", return_value={"vision_service": object()}),
                patch("app.task_card_model_test_worker.RealVisionClient", FakeVisionClient),
                patch("app.task_card_model_test_worker.recognize_task_card_with_diagnostics", return_value={
                    "model_result": model_result,
                    "raw_response": '{"success":true}',
                    "model": "qwen-test-model",
                    "provider_request_id": "provider-1",
                    "elapsed_ms": 12.3,
                }),
            ):
                worker.run()
            self.assertEqual(failures, [])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["success"])
            self.assertTrue(results[0]["recognition_success"])
            self.assertEqual(results[0]["validated_result"]["scene_description"], "模型测试场景")
            self.assertIn("未进入比赛会话", results[0]["validation_message"])


if __name__ == "__main__":
    unittest.main()
