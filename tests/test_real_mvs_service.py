from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.config import Endpoint
from app.vision_client import RealVisionClient
from vision.mvs_camera import MvsCapture, MvsDevice, sdk_files_available
from vision.real_mvs_service import RealMvsVisionService
from vision.workspace_localizer import estimate_six_color_area_range


def _profile(*, detector: bool) -> dict:
    value = {
        "approved": True, "exposure_us": 5000.0, "gain": 1.0,
        "white_balance": {"red": 1.1, "green": 1.0, "blue": 1.2},
        "roi": {"width": 200, "height": 200, "offset_x": 0, "offset_y": 0},
        "trigger_mode": "software", "requires_nine_point_calibration": detector,
    }
    if detector:
        value["detector"] = {
            "approved": True, "roi": [0, 0, 200, 200],
            "confidence_min": 0.6, "min_area_px": 100.0, "max_area_px": 10000.0,
            "hsv_ranges": {
                "红": [{"lower": [0, 120, 70], "upper": [10, 255, 255]}],
                "橙": [{"lower": [11, 100, 80], "upper": [25, 255, 255]}],
                "黄": [{"lower": [26, 100, 80], "upper": [35, 255, 255]}],
                "绿": [{"lower": [36, 80, 60], "upper": [85, 255, 255]}],
                "蓝": [{"lower": [86, 80, 60], "upper": [130, 255, 255]}],
                "紫": [{"lower": [131, 60, 60], "upper": [169, 255, 255]}],
            },
        }
    return value


class FakeCamera:
    def __init__(self) -> None:
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (75, 75), (125, 125), (0, 0, 255), -1)
        self.closed = False
        self.capture_count = 0

    def open_first_available(self) -> MvsDevice:
        return MvsDevice(0, "TEST-CAMERA", "USB3")

    def capture(self, profile: dict, *, require_approved: bool = True) -> MvsCapture:
        self.capture_count += 1
        return MvsCapture(self.image.copy(), 7, True, {
            "exposure_us": profile["exposure_us"], "gain": profile["gain"],
            "white_balance": profile["white_balance"], "roi": profile["roi"], "trigger_mode": "software",
        })

    def close(self) -> None:
        self.closed = True


class RealMvsServiceTest(unittest.TestCase):
    def test_png_save_supports_unicode_windows_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            service = RealMvsVisionService.__new__(RealMvsVisionService)
            service.session_root = (Path(temp) / "中文证据目录").resolve()
            service.session_root.mkdir(parents=True)
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            path = service._save_image("会话-1", "blocks", "request-1", image)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.calibration_root = root / "calibration"
        self.session_root = root / "sessions"
        self.camera_config = {
            "schema_version": 1, "sdk_family": "hikrobot_mvs",
            "mounting": "eye_in_hand", "fresh_frame_max_age_ms": 1000,
            "profiles": {"task_card": _profile(detector=False), "blocks": _profile(detector=True), "trays": _profile(detector=True)},
        }
        self.camera = FakeCamera()
        self.service = RealMvsVisionService(
            camera=self.camera, camera_config=self.camera_config,
            endpoint=Endpoint("vision_service", "127.0.0.1", 9001, "outbound", "real_mvs_vision", {}),
            calibration_root=self.calibration_root, session_root=self.session_root,
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temp.cleanup()

    def _write_calibration(self, scene: str) -> str:
        calibration_id = f"CAL-{scene.upper()}-CLIENT"
        directory = self.calibration_root / scene
        directory.mkdir(parents=True, exist_ok=True)
        calibration = {
            "schema_version": 1, "scene": scene, "data_origin": "camera_vision",
            "usable_for_real_robot": True, "approved": True,
            "active_tcp": "TCP-REAL", "calibration_id": calibration_id,
            "photo_point": f"{scene}_photo", "image_width": 200, "image_height": 200,
            "homography_pixel_to_tool_mm": [[0.1, 0.0, -10.0], [0.0, 0.1, -10.0], [0.0, 0.0, 1.0]],
            "reference_detections": {
                color: {"pixel_u": 100.0, "pixel_v": 100.0, "r_image_deg": 0.0, "confidence": 1.0}
                for color in ("红", "橙", "黄", "绿", "蓝", "紫")
            },
        }
        (directory / "approved.json").write_text(json.dumps(calibration, ensure_ascii=False), encoding="utf-8")
        return calibration_id

    def _direct_client(self, calibration_ids: dict[str, str] | None = None) -> RealVisionClient:
        client = RealVisionClient(
            self.service.endpoint, active_tcp="TCP-REAL",
            calibration_ids=calibration_ids, fresh_frame_max_age_ms=1000,
        )
        client._exchange = self.service.handle  # type: ignore[method-assign]
        return client

    def test_sdk_check_does_not_require_camera_enumeration(self) -> None:
        self.assertTrue(sdk_files_available())

    def test_six_color_area_range_is_estimated_without_existing_area_limits(self) -> None:
        hsv = np.zeros((200, 300, 3), dtype=np.uint8)
        for index, hue in enumerate((0, 18, 30, 60, 105, 150)):
            x = 10 + index * 45
            cv2.rectangle(hsv, (x, 70), (x + 25, 95), (hue, 230, 230), -1)
        # Tiny square-shaped color noise must not outrank a real target merely
        # because its aspect ratio and rectangularity are both perfect.
        cv2.rectangle(hsv, (2, 2), (5, 5), (18, 230, 230), -1)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        profile = _profile(detector=True)
        profile["detector"]["min_area_px"] = "UNSET"
        profile["detector"]["max_area_px"] = "UNSET"
        result = estimate_six_color_area_range(image, profile=profile)
        self.assertEqual(set(result["areas_px"]), {"红", "橙", "黄", "绿", "蓝", "紫"})
        self.assertLess(result["min_area_px"], min(result["areas_px"].values()))
        self.assertGreater(result["max_area_px"], max(result["areas_px"].values()))

    def test_health_and_fresh_task_card_capture(self) -> None:
        health = self.service.handle({"type": "health_request", "protocol_version": 1, "expected_service": "real_mvs_vision"})
        self.assertEqual(health["status"], "ready")
        result = self.service.handle({
            "type": "capture_frame", "protocol_version": 1, "request_id": "CARD-1",
            "session_id": "SESSION-1", "scene": "task_card", "profile": "task_card",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["data_origin"], "camera_vision")
        self.assertTrue(Path(result["image_path"]).is_file())
        self.assertTrue(Path(result["image_path"]).resolve().is_relative_to(self.session_root.resolve()))

    def test_real_service_task_card_response_is_accepted_by_real_client(self) -> None:
        session_id = "CLIENT-CARD-SESSION"
        frame = self._direct_client().capture_task_card(
            request_id="CLIENT-CARD-1", session_id=session_id,
            session_dir=self.session_root / session_id,
        )
        self.assertTrue(frame.image_path.is_file())
        self.assertEqual(frame.request_id, "CLIENT-CARD-1")

    def test_real_service_block_response_is_accepted_by_real_client(self) -> None:
        calibration_id = self._write_calibration("blocks")
        detection = self._direct_client({"blocks": calibration_id}).locate_block(
            request_id="CLIENT-BLOCK-1", color="红", photo_point="blocks_photo",
            session_id="CLIENT-BLOCK-SESSION",
        )
        self.assertEqual(detection["color"], "红")

    def test_formal_block_uses_largest_area_even_when_shape_is_only_a_warning(self) -> None:
        calibration_id = self._write_calibration("blocks")
        hsv = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(hsv, (20, 30), (40, 50), (0, 230, 230), -1)
        cv2.rectangle(hsv, (75, 70), (165, 120), (0, 230, 230), -1)
        self.camera.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        detection = self._direct_client({"blocks": calibration_id}).locate_block(
            request_id="CLIENT-BLOCK-LARGEST", color="红", photo_point="blocks_photo",
            session_id="CLIENT-BLOCK-LARGEST-SESSION",
        )
        self.assertGreater(detection["dx_tool_m"], 0.0)

    def test_real_service_trays_response_is_accepted_by_real_client(self) -> None:
        calibration_id = self._write_calibration("trays")
        hsv = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = ((30, 30), (90, 30), (150, 30), (30, 110), (90, 110), (150, 110))
        for (x, y), hue in zip(positions, (0, 18, 30, 60, 105, 150)):
            cv2.rectangle(hsv, (x - 12, y - 12), (x + 12, y + 12), (hue, 230, 230), -1)
        self.camera.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        detections = self._direct_client({"trays": calibration_id}).locate_trays(
            request_id="CLIENT-TRAYS-1", photo_point="trays_photo",
            session_id="CLIENT-TRAYS-SESSION",
        )
        self.assertEqual(set(detections), {"红", "橙", "黄", "绿", "蓝", "紫"})

    def test_real_service_partial_trays_response_is_accepted_by_real_client(self) -> None:
        calibration_id = self._write_calibration("trays")
        hsv = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = ((30, 30), (90, 30), (150, 30), (30, 110), (90, 110))
        for (x, y), hue in zip(positions, (0, 18, 30, 60, 105)):
            cv2.rectangle(hsv, (x - 12, y - 12), (x + 12, y + 12), (hue, 230, 230), -1)
        self.camera.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        detections = self._direct_client({"trays": calibration_id}).locate_trays(
            request_id="CLIENT-TRAYS-PARTIAL", photo_point="trays_photo",
            session_id="CLIENT-TRAYS-PARTIAL-SESSION",
        )
        self.assertEqual(set(detections), {"红", "橙", "黄", "绿", "蓝"})

    def test_unapproved_profile_can_only_be_true_readback_validated(self) -> None:
        self.camera_config["profiles"]["task_card"]["approved"] = False
        result = self.service.handle({
            "type": "profile_validate", "protocol_version": 1, "request_id": "PROFILE-TASK",
            "session_id": "PROFILE-SESSION", "profile": "task_card",
        })
        self.assertTrue(result["success"], result)
        self.assertTrue(result["parameters_applied"])
        self.assertEqual(result["configured_parameters"]["roi"]["width"], 200)
        self.assertEqual(result["frame_number"], 7)

    def test_manual_scene_capture_saves_raw_block_image_without_detector_or_calibration(self) -> None:
        self.camera_config["profiles"]["blocks"]["approved"] = False
        result = self.service.handle({
            "type": "manual_scene_capture", "protocol_version": 1,
            "request_id": "blocks-20260814-160000-000001",
            "session_id": "manual-blocks-20260814-160000-000000",
            "scene": "blocks",
        })
        self.assertTrue(result["success"], result)
        path = Path(result["image_path"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.name, "blocks")
        self.assertEqual(path.parent.parent.name, "manual-captures")
        self.assertTrue(result["parameters_applied"])
        self.assertNotIn("annotated_image_path", result)

    def test_manual_scene_recognition_uses_one_new_frame_and_returns_pixels_without_calibration(self) -> None:
        before = self.camera.capture_count
        result = self._direct_client().recognize_manual_scene(
            request_id="blocks-recognize-1", session_id="manual-recognize-blocks-1", scene="blocks",
        )
        self.assertEqual(self.camera.capture_count, before + 1)
        self.assertFalse(result["calibration_available"])
        self.assertEqual([item["color"] for item in result["detections"]], ["红"])
        self.assertEqual(set(result["missing_colors"]), {"橙", "黄", "绿", "蓝", "紫"})
        self.assertIn("pixel_u", result["detections"][0])
        self.assertNotIn("delta_x_tool_m", result["detections"][0])
        self.assertTrue(Path(result["image_path"]).is_file())
        self.assertTrue(Path(result["annotated_image_path"]).is_file())

    def test_manual_scene_recognition_uses_current_calibration_for_all_six_colors(self) -> None:
        calibration_id = self._write_calibration("blocks")
        hsv = np.zeros((200, 200, 3), dtype=np.uint8)
        positions = ((30, 30), (90, 30), (150, 30), (30, 110), (90, 110), (150, 110))
        for (x, y), hue in zip(positions, (0, 18, 30, 60, 105, 150)):
            cv2.rectangle(hsv, (x - 12, y - 12), (x + 12, y + 12), (hue, 230, 230), -1)
        self.camera.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = self._direct_client().recognize_manual_scene(
            request_id="blocks-recognize-2", session_id="manual-recognize-blocks-2", scene="blocks",
        )
        self.assertTrue(result["calibration_available"])
        self.assertEqual(result["calibration_id"], calibration_id)
        self.assertEqual(result["missing_colors"], [])
        self.assertEqual({item["color"] for item in result["detections"]}, {"红", "橙", "黄", "绿", "蓝", "紫"})
        self.assertTrue(all("delta_x_tool_m" in item and "delta_r_rad" in item for item in result["detections"]))

    def test_service_can_start_for_profile_readback_before_detector_is_complete(self) -> None:
        self.camera_config["profiles"]["blocks"]["detector"]["roi"] = "UNSET"
        camera = FakeCamera()
        service = RealMvsVisionService(
            camera=camera, camera_config=self.camera_config,
            endpoint=Endpoint("vision_service", "127.0.0.1", 9001, "outbound", "real_mvs_vision", {}),
            calibration_root=self.calibration_root, session_root=self.session_root,
        )
        service.close()
        self.assertTrue(camera.closed)

    def test_block_location_uses_matching_real_calibration(self) -> None:
        directory = self.calibration_root / "blocks"
        directory.mkdir(parents=True)
        calibration = {
            "schema_version": 1, "scene": "blocks", "data_origin": "camera_vision",
            "usable_for_real_robot": True, "approved": True,
            "active_tcp": "TCP-REAL", "calibration_id": "CAL-BLOCKS-1",
            "photo_point": "blocks_photo", "image_width": 200, "image_height": 200,
            "homography_pixel_to_tool_mm": [[0.1, 0.0, -10.0], [0.0, 0.1, -10.0], [0.0, 0.0, 1.0]],
            "reference_detections": {
                color: {"pixel_u": 100.0, "pixel_v": 100.0, "r_image_deg": 0.0, "confidence": 1.0}
                for color in ("红", "橙", "黄", "绿", "蓝", "紫")
            },
        }
        (directory / "approved.json").write_text(json.dumps(calibration, ensure_ascii=False), encoding="utf-8")
        result = self.service.handle({
            "type": "capture_and_locate", "protocol_version": 1, "request_id": "BLOCK-1",
            "session_id": "SESSION-2", "scene": "blocks", "target_color": "红",
            "photo_point": "blocks_photo",
            "calibration_id": "CAL-BLOCKS-1", "active_tcp": "TCP-REAL",
        })
        self.assertTrue(result["success"], result)
        self.assertEqual(result["detection"]["color"], "红")
        self.assertAlmostEqual(result["detection"]["dx_tool_m"], 0.0, delta=0.002)
        self.assertAlmostEqual(result["detection"]["dy_tool_m"], 0.0, delta=0.002)
        self.assertTrue(Path(result["image_path"]).is_file())
        self.assertTrue(Path(result["annotated_image_path"]).is_file())
        self.assertTrue(result["detection_summary"]["success"])

    def test_failed_color_detection_still_returns_annotated_evidence(self) -> None:
        self.camera.image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.service.handle({
            "type": "detector_validate", "protocol_version": 1,
            "request_id": "DETECT-FAIL-1", "session_id": "DETECT-FAIL-SESSION",
            "scene": "blocks",
        })
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "TARGET_NOT_FOUND")
        self.assertTrue(Path(result["image_path"]).is_file())
        self.assertTrue(Path(result["annotated_image_path"]).is_file())
        self.assertFalse(result["detection_summary"]["success"])
        self.assertEqual(len(result["detection_summary"]["colors"]), 6)

    def test_real_nine_point_session_writes_unapproved_candidate(self) -> None:
        self.service.profiles["blocks"]["detector"]["approved"] = False
        self.service.profiles["blocks"]["detector"]["min_area_px"] = "UNSET"
        self.service.profiles["blocks"]["detector"]["max_area_px"] = "UNSET"
        begin = self.service.handle({
            "type": "calibration_begin", "protocol_version": 1, "request_id": "N9-BEGIN", "session_id": "N9-SESSION",
            "scene": "blocks", "profile": "blocks", "target_color": "红", "photo_point": "blocks_photo",
            "robot_serial": "ROBOT-1", "active_tcp": "TCP-REAL", "step_x_mm": 20.0, "step_y_mm": 15.0,
        })
        self.assertTrue(begin["success"], begin)
        tools = [(20, 15), (0, 15), (-20, 15), (-20, 0), (0, 0), (20, 0), (20, -15), (0, -15), (-20, -15)]
        for index, (tool_x, tool_y) in enumerate(tools, start=1):
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            u, v = int(100 + 2 * tool_x), int(100 + 2 * tool_y)
            cv2.rectangle(image, (u - 10, v - 10), (u + 10, v + 10), (0, 0, 255), -1)
            if index == 5:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                for (x, y), hue in zip(((25, 25), (55, 25), (145, 25), (25, 165), (165, 165)), (18, 30, 60, 105, 150)):
                    cv2.rectangle(hsv, (x - 8, y - 8), (x + 8, y + 8), (hue, 230, 230), -1)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.camera.image = image
            point = self.service.handle({
                "type": "calibration_capture_point", "protocol_version": 1, "request_id": f"N9-P{index}", "session_id": "N9-SESSION",
                "index": index, "actual_tcp_pose": [0, 0, 0, 0, 0, 0],
                "tool_x_mm": tool_x, "tool_y_mm": tool_y,
            })
            self.assertTrue(point["success"], point)
        self.service.profiles["blocks"]["detector"]["min_area_px"] = 100.0
        self.service.profiles["blocks"]["detector"]["max_area_px"] = 10000.0
        finish = self.service.handle({"type": "calibration_finish", "protocol_version": 1, "request_id": "N9-FINISH", "session_id": "N9-SESSION"})
        self.assertTrue(finish["success"], finish); self.assertFalse(finish["approved"])
        candidate = json.loads(Path(finish["candidate_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(candidate["samples"]), 9); self.assertFalse(candidate["usable_for_real_robot"])

    def test_new_calibration_begin_replaces_stale_unfinished_session(self) -> None:
        def begin(session_id: str) -> dict:
            return self.service.handle({
                "type": "calibration_begin", "protocol_version": 1,
                "request_id": f"{session_id}-BEGIN", "session_id": session_id,
                "scene": "blocks", "profile": "blocks", "target_color": "红",
                "photo_point": "blocks_photo",
                "robot_serial": "ROBOT-1", "active_tcp": "TCP-REAL",
                "step_x_mm": 10.0, "step_y_mm": 10.0,
            })

        first = begin("N9-OLD")
        self.assertTrue(first["success"], first)
        second = begin("N9-NEW")
        self.assertTrue(second["success"], second)
        self.assertEqual(second["replaced_session_id"], "N9-OLD")
        health = self.service.handle({
            "type": "health_request", "protocol_version": 1,
            "expected_service": "real_mvs_vision",
        })
        self.assertEqual(health["calibration_session"], "N9-NEW")
        stale_point = self.service.handle({
            "type": "calibration_capture_point", "protocol_version": 1,
            "request_id": "OLD-P1", "session_id": "N9-OLD",
            "index": 1,
            "actual_tcp_pose": [0, 0, 0, 0, 0, 0],
            "tool_x_mm": 10.0, "tool_y_mm": 10.0,
        })
        self.assertFalse(stale_point["success"])


if __name__ == "__main__":
    unittest.main()
