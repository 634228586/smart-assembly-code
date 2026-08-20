from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.nine_point import NinePointError, approve_candidate, approve_candidate_without_direction_validation, build_candidate, build_grid, fit_pixel_to_tool, write_candidate
from app.paths import PACKAGE_ROOT, resolve_project_path


class NinePointTest(unittest.TestCase):
    @staticmethod
    def _references() -> dict[str, dict[str, float]]:
        return {
            color: {"pixel_u": 100.0 + index * 10.0, "pixel_v": 100.0, "r_image_deg": 0.0, "confidence": 1.0}
            for index, color in enumerate(("红", "橙", "黄", "绿", "蓝", "紫"))
        }

    @staticmethod
    def _samples(noise_index: int | None = None) -> list[dict[str, float | int]]:
        samples = []
        for point in build_grid(20, 15):
            tool_x, tool_y = point.expected_tool_x_mm, point.expected_tool_y_mm
            u, v = 100 + 2 * tool_x, 100 + 2 * tool_y
            if noise_index == point.index: u += 5
            samples.append({"index": point.index, "pixel_u": u, "pixel_v": v, "tool_x_mm": tool_x, "tool_y_mm": tool_y})
        return samples

    def test_fit_and_approve_latest_candidate(self) -> None:
        samples = self._samples(); fit = fit_pixel_to_tool(samples)
        self.assertLess(fit.max_error_mm, 1e-6)
        candidate = build_candidate(
            scene="blocks", robot_serial="ROB1", active_tcp="TCP1",
            photo_point="blocks_photo", image_width=400, image_height=300, target_color="红",
            step_x_mm=20, step_y_mm=15, samples=samples, fit=fit,
            reference_detections=self._references(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "candidate.json"; active = root / "real" / "9point_blocks.json"
            write_candidate(source, candidate)
            approved = approve_candidate(source, active, root / "archive", {
                "x_positive": {"dx_mm": 10.0, "dy_mm": 0.1},
                "y_positive": {"dx_mm": -0.1, "dy_mm": 10.0},
                "angle_zero": 0.5, "angle_positive_10deg": 10.2,
            })
            self.assertTrue(approved["approved"]); self.assertTrue(active.is_file())
            self.assertTrue(json.loads(active.read_text(encoding="utf-8"))["usable_for_real_robot"])

    def test_bad_point_is_not_silently_removed(self) -> None:
        with self.assertRaises(NinePointError):
            fit_pixel_to_tool(self._samples(noise_index=3))

    def test_operator_can_activate_candidate_without_direction_validation(self) -> None:
        samples = self._samples(); fit = fit_pixel_to_tool(samples)
        candidate = build_candidate(
            scene="blocks", robot_serial="ROB1", active_tcp="TCP1",
            photo_point="blocks_photo", image_width=400, image_height=300, target_color="红",
            step_x_mm=20, step_y_mm=15, samples=samples, fit=fit,
            reference_detections=self._references(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "candidate.json"; active = root / "real" / "9point_blocks.json"
            write_candidate(source, candidate)
            approved = approve_candidate_without_direction_validation(source, active, root / "archive")
            self.assertTrue(approved["approved"])
            self.assertEqual(approved["approval_source"], "operator_skipped_direction_validation")
            self.assertTrue(approved["direction_validation"]["skipped"])

    def test_candidate_stores_project_evidence_paths_as_relative(self) -> None:
        samples = self._samples()
        evidence = PACKAGE_ROOT / "data" / "sessions" / "test-nine-point" / "calibration" / "blocks"
        for index, sample in enumerate(samples, 1):
            sample["image_path"] = str(evidence / f"point-{index}-annotated.png")
            sample["raw_image_path"] = str(evidence / f"point-{index}.png")
            sample["annotated_image_path"] = str(evidence / f"point-{index}-annotated.png")
        candidate = build_candidate(
            scene="blocks", robot_serial="ROB1", active_tcp="TCP1",
            photo_point="blocks_photo", image_width=400, image_height=300, target_color="红",
            step_x_mm=20, step_y_mm=15, samples=samples, fit=fit_pixel_to_tool(samples),
            reference_detections=self._references(),
        )
        first = candidate["samples"][0]
        self.assertEqual(first["raw_image_path"], "data/sessions/test-nine-point/calibration/blocks/point-1.png")
        self.assertFalse(Path(first["image_path"]).is_absolute())

    def test_legacy_absolute_evidence_path_rebases_to_current_project(self) -> None:
        legacy = r"C:\Users\dqq\Desktop\111\装配赛正式代码\data\sessions\old-session\calibration\blocks\point-1.png"
        self.assertEqual(
            resolve_project_path(legacy),
            (PACKAGE_ROOT / "data" / "sessions" / "old-session" / "calibration" / "blocks" / "point-1.png").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
