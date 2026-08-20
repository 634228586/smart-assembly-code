from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.camera_config_editor import (
    CameraConfigInputError, PROFILE_KEYS, approve_detector_values, approve_profile_batch, approve_profile_batch_by_operator, approve_profile_values,
    load_camera_editor_values, save_camera_editor_values, save_detector_editor_values,
)


class CameraConfigEditorTest(unittest.TestCase):
    def _file(self, root: Path) -> Path:
        path = root / "camera.json"
        profiles = {}
        for key in PROFILE_KEYS:
            profiles[key] = {
                "approved": True, "exposure_us": "UNSET", "gain": "UNSET",
                "white_balance": {"red": "UNSET", "green": "UNSET", "blue": "UNSET"},
                "roi": {"width": "UNSET", "height": "UNSET", "offset_x": "UNSET", "offset_y": "UNSET"},
                "trigger_mode": "software", "requires_nine_point_calibration": key != "task_card",
            }
            if key != "task_card": profiles[key]["detector"] = {"approved": True, "keep": 7}
        path.write_text(json.dumps({
            "schema_version": 1, "sdk_family": "hikrobot_mvs",
            "mounting": "eye_in_hand", "fresh_frame_max_age_ms": 1000, "profiles": profiles,
        }), encoding="utf-8")
        return path

    @staticmethod
    def _values() -> dict[str, dict[str, object]]:
        return {key: {
            "exposure_us": 12000, "gain": 3.5, "white_red": 2097, "white_green": 1024, "white_blue": 1559,
            "width": 1920, "height": 1080, "offset_x": 0, "offset_y": 0,
        } for key in PROFILE_KEYS}

    def test_save_preserves_structure_and_revokes_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp))
            save_camera_editor_values(path, profile_values=self._values())
            camera = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(camera["mounting"], "eye_in_hand")
            self.assertEqual(camera["profiles"]["blocks"]["roi"]["width"], 1920)
            self.assertFalse(camera["profiles"]["task_card"]["approved"])
            self.assertFalse(camera["profiles"]["blocks"]["detector"]["approved"])
            self.assertEqual(camera["profiles"]["blocks"]["detector"]["roi"], [0, 0, 1920, 1080])
            self.assertEqual(camera["profiles"]["blocks"]["detector"]["keep"], 7)
            profiles = load_camera_editor_values(path)
            self.assertEqual(profiles["trays"]["height"], 1080)

    def test_invalid_or_incomplete_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp)); values = self._values()
            values["task_card"]["width"] = "1.5"
            with self.assertRaises(CameraConfigInputError):
                save_camera_editor_values(path, profile_values=values)

    def test_detector_requires_six_colors_and_exact_validated_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp))
            save_camera_editor_values(path, profile_values=self._values())
            hsv = {
                color: [{"lower": [index * 10, 60, 60], "upper": [index * 10 + 9, 255, 255]}]
                for index, color in enumerate(("红", "橙", "黄", "绿", "蓝", "紫"))
            }
            save_detector_editor_values(
                path, scene="blocks", roi_values=[100, 100, 200, 200], confidence_min=0.6,
                min_area_px=100, max_area_px=10000, hsv_json=json.dumps(hsv, ensure_ascii=False),
            )
            camera = json.loads(path.read_text(encoding="utf-8")); detector = camera["profiles"]["blocks"]["detector"]
            self.assertEqual(detector["roi"], [0, 0, 1920, 1080])
            import hashlib
            digest = hashlib.sha256(json.dumps(detector, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            approve_detector_values(path, scene="blocks", expected_sha256=digest)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["profiles"]["blocks"]["detector"]["approved"])
            changed = save_detector_editor_values(
                path, scene="blocks", roi_values=[0, 0, 1920, 1080], confidence_min=0.6,
                min_area_px=100, max_area_px=10000, hsv_json=json.dumps(hsv, ensure_ascii=False),
            )
            self.assertFalse(changed)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["profiles"]["blocks"]["detector"]["approved"])
            changed = save_detector_editor_values(
                path, scene="blocks", roi_values=[0, 0, 1920, 1080], confidence_min=0.6,
                min_area_px=100, max_area_px=10001, hsv_json=json.dumps(hsv, ensure_ascii=False),
            )
            self.assertTrue(changed)
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["profiles"]["blocks"]["detector"]["approved"])
            with self.assertRaises(CameraConfigInputError):
                approve_detector_values(path, scene="blocks", expected_sha256=digest)

    def test_profile_approval_requires_exact_true_readback_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp))
            save_camera_editor_values(path, profile_values=self._values())
            camera = json.loads(path.read_text(encoding="utf-8")); profile = camera["profiles"]["task_card"]
            import hashlib
            digest = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            approve_profile_values(path, profile_name="task_card", expected_sha256=digest)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["profiles"]["task_card"]["approved"])
            with self.assertRaises(CameraConfigInputError):
                approve_profile_values(path, profile_name="task_card", expected_sha256=digest)

    def test_operator_can_approve_complete_profiles_without_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp))
            save_camera_editor_values(path, profile_values=self._values())
            approve_profile_batch_by_operator(path)
            camera = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(all(camera["profiles"][name]["approved"] is True for name in PROFILE_KEYS))
            self.assertTrue(all(
                camera["profiles"][name]["approval_source"] == "operator_confirmed_without_preflight_readback"
                for name in PROFILE_KEYS
            ))

    def test_three_profile_approval_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._file(Path(temp))
            save_camera_editor_values(path, profile_values=self._values())
            camera = json.loads(path.read_text(encoding="utf-8"))
            import hashlib
            hashes = {
                name: hashlib.sha256(json.dumps(camera["profiles"][name], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
                for name in PROFILE_KEYS
            }
            hashes["trays"] = "wrong"
            with self.assertRaises(CameraConfigInputError):
                approve_profile_batch(path, expected_sha256_by_profile=hashes)
            after = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(all(after["profiles"][name]["approved"] is False for name in PROFILE_KEYS))


if __name__ == "__main__":
    unittest.main()
