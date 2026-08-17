from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.protocols import ProtocolError, normalize_speech_health, validate_recognition_result, validate_service_identity, validate_workspace_result


COLORS = ["红", "橙", "黄", "绿", "蓝", "紫"]


class ProtocolTest(unittest.TestCase):
    @staticmethod
    def _detection(color: str = "红") -> dict[str, float | str]:
        return {
            "color": color,
            "dx_tool_m": 0.01, "dy_tool_m": -0.01, "r_image_rad": 0.1, "confidence": 0.95,
            "delta_x_tool_m": 0.001, "delta_y_tool_m": -0.002, "delta_r_rad": 0.01,
            "reference_pixel_u": 100.0, "reference_pixel_v": 101.0,
            "current_pixel_u": 102.0, "current_pixel_v": 103.0,
        }

    def _task2(self, session: Path) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        request_id = "real-session-task-2"
        image = session / "task_card" / "capture.png"
        image.parent.mkdir(parents=True); image.write_bytes(b"not-decoded-in-protocol-test")
        return {
            "schema_version": 1, "type": "recognition_result", "request_id": request_id,
            "task_type": "task_2", "success": True, "confidence": 0.99, "recognized_at": now,
            "raw_text": "任务卡二", "source_image": {
                "image_id": "capture", "path": str(image.resolve()), "captured_at": now,
                "camera_serial": "MVS-REAL-001", "capture_request_id": request_id,
            },
            "sequence": [
                {"order": index + 1, "block_color": color, "tray_color": COLORS[(index + 1) % 6]}
                for index, color in enumerate(COLORS)
            ],
        }

    def test_task2_requires_current_session_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            result = validate_recognition_result(self._task2(session), session_dir=session, camera_serial="MVS-REAL-001")
            self.assertEqual(len(result["sequence"]), 6)

    def test_wrong_camera_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp); payload = self._task2(session)
            with self.assertRaises(ProtocolError):
                validate_recognition_result(payload, session_dir=session, camera_serial="OTHER")

    def test_missing_task_card_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp); payload = self._task2(session)
            Path(payload["source_image"]["path"]).unlink()
            with self.assertRaises(ProtocolError):
                validate_recognition_result(payload, session_dir=session, camera_serial="MVS-REAL-001")

    def test_service_identity_is_required(self) -> None:
        with self.assertRaises(ProtocolError):
            validate_service_identity({"service": "wrong", "protocol_version": 1, "status": "ready"}, expected_service="real_mvs_vision")

    def test_real_arm_speech_health_is_strictly_adapted(self) -> None:
        payload = {
            "ready": True, "service": "arm_speech_service", "version": "m28-4-real", "provider": "real",
            "capture_ready": "lazy", "tts_backends": [{"name": "piper", "mode": "offline"}],
            "models": {"asr_exists": True, "keyword_exists": True, "vad_exists": True, "piper_exists": True},
        }
        result = normalize_speech_health(payload, expected_service="arm_speech_service")
        self.assertEqual(result["protocol_version"], 1)
        self.assertEqual(set(result["capabilities"]), {"asr", "tts"})
        payload["models"]["asr_exists"] = False
        with self.assertRaises(ProtocolError):
            normalize_speech_health(payload, expected_service="arm_speech_service")

    def test_workspace_source_and_identity_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.png"; raw_path.write_bytes(b"raw")
            annotated_path = Path(temp) / "annotated.png"; annotated_path.write_bytes(b"annotated")
            payload = {
                "service": "real_mvs_vision", "protocol_version": 1, "request_id": "r1",
                "scene": "blocks", "data_origin": "camera_vision", "camera_serial": "MVS-REAL-001",
                "calibration_id": "CAL-1", "active_tcp": "TCP-1",
                "coordinate_frame": "active_tool_at_photo_pose", "success": True,
                "captured_at": datetime.now(timezone.utc).isoformat(), "target_color": "红",
                "frame_number": 7, "image_width": 200, "image_height": 200,
                "configured_parameters": {"roi": {"width": 200, "height": 200, "offset_x": 0, "offset_y": 0}},
                "image_path": str(raw_path), "annotated_image_path": str(annotated_path),
                "detection_summary": {"success": True, "colors": [{"color": "红", "status": "success"}]},
                "detection": self._detection(),
            }
            payload["photo_point"] = "blocks_photo"
            validate_workspace_result(payload, scene="blocks", request_id="r1", camera_serial="MVS-REAL-001", calibration_id="CAL-1", active_tcp="TCP-1", photo_point="blocks_photo")
            payload["data_origin"] = "not-real"
            with self.assertRaises(ProtocolError):
                validate_workspace_result(payload, scene="blocks", request_id="r1", camera_serial="MVS-REAL-001", calibration_id="CAL-1", active_tcp="TCP-1", photo_point="blocks_photo")

    def test_workspace_old_frame_is_rejected(self) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "service": "real_mvs_vision", "protocol_version": 1, "request_id": "r-old",
            "scene": "blocks", "data_origin": "camera_vision", "camera_serial": "MVS-REAL-001",
            "calibration_id": "CAL-1", "active_tcp": "TCP-1",
            "coordinate_frame": "active_tool_at_photo_pose", "success": True,
            "captured_at": (now - timedelta(seconds=3)).isoformat(), "target_color": "红",
            "frame_number": 7, "image_width": 200, "image_height": 200,
            "configured_parameters": {"roi": {"width": 200, "height": 200, "offset_x": 0, "offset_y": 0}},
            "detection": {"color": "红", "dx_tool_m": 0.01, "dy_tool_m": -0.01, "r_image_rad": 0.1, "confidence": 0.95},
        }
        with self.assertRaises(ProtocolError):
            payload["photo_point"] = "blocks_photo"
            validate_workspace_result(payload, scene="blocks", request_id="r-old", camera_serial="MVS-REAL-001", calibration_id="CAL-1", active_tcp="TCP-1", photo_point="blocks_photo", now=now, fresh_frame_max_age_ms=1000)

    def test_workspace_post_capture_processing_within_five_seconds_is_accepted(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.png"; raw_path.write_bytes(b"raw")
            annotated_path = Path(temp) / "annotated.png"; annotated_path.write_bytes(b"annotated")
            payload = {
                "service": "real_mvs_vision", "protocol_version": 1, "request_id": "r-slow",
                "scene": "blocks", "data_origin": "camera_vision", "camera_serial": "MVS-REAL-001",
                "calibration_id": "CAL-1", "active_tcp": "TCP-1", "photo_point": "blocks_photo",
                "coordinate_frame": "active_tool_at_photo_pose", "success": True,
                "captured_at": (now - timedelta(seconds=3)).isoformat(), "target_color": COLORS[0],
                "frame_number": 7, "image_width": 3072, "image_height": 2048,
                "configured_parameters": {"roi": {"width": 3072, "height": 2048, "offset_x": 0, "offset_y": 0}},
                "image_path": str(raw_path), "annotated_image_path": str(annotated_path),
                "detection_summary": {"success": True, "colors": [{"color": COLORS[0], "status": "success"}]},
                "detection": self._detection(COLORS[0]),
            }

            result = validate_workspace_result(
                payload, scene="blocks", request_id="r-slow", camera_serial="MVS-REAL-001",
                calibration_id="CAL-1", active_tcp="TCP-1", photo_point="blocks_photo",
                request_started_at=now - timedelta(seconds=4), now=now, fresh_frame_max_age_ms=5000,
            )

            self.assertEqual(result["frame_number"], 7)

    def test_workspace_frame_before_request_is_rejected_inside_five_second_window(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.png"; raw_path.write_bytes(b"raw")
            annotated_path = Path(temp) / "annotated.png"; annotated_path.write_bytes(b"annotated")
            payload = {
                "service": "real_mvs_vision", "protocol_version": 1, "request_id": "r-stale",
                "scene": "blocks", "data_origin": "camera_vision", "camera_serial": "MVS-REAL-001",
                "calibration_id": "CAL-1", "active_tcp": "TCP-1", "photo_point": "blocks_photo",
                "coordinate_frame": "active_tool_at_photo_pose", "success": True,
                "captured_at": (now - timedelta(seconds=1)).isoformat(), "target_color": COLORS[0],
                "frame_number": 7, "image_width": 3072, "image_height": 2048,
                "configured_parameters": {"roi": {"width": 3072, "height": 2048, "offset_x": 0, "offset_y": 0}},
                "image_path": str(raw_path), "annotated_image_path": str(annotated_path),
                "detection_summary": {"success": True, "colors": [{"color": COLORS[0], "status": "success"}]},
                "detection": {"color": COLORS[0], "dx_tool_m": 0.01, "dy_tool_m": -0.01, "r_image_rad": 0.1, "confidence": 0.95},
            }

            with self.assertRaisesRegex(ProtocolError, "早于本次拍照请求"):
                validate_workspace_result(
                    payload, scene="blocks", request_id="r-stale", camera_serial="MVS-REAL-001",
                    calibration_id="CAL-1", active_tcp="TCP-1", photo_point="blocks_photo",
                    request_started_at=now - timedelta(milliseconds=500), now=now, fresh_frame_max_age_ms=5000,
                )


if __name__ == "__main__":
    unittest.main()
