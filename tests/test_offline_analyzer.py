from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision.offline_analyzer import (
    OfflineAnalysisError,
    analyze_image,
    discover_images,
    normalize_detector_config,
    write_candidate,
)
from vision.workspace_localizer import COLORS, analyze_color_candidates


HUES = dict(zip(COLORS, (0, 15, 30, 60, 110, 145)))


def detector() -> dict:
    return {
        "roi": [0, 0, 360, 240],
        "confidence_min": 0.6,
        "min_area_px": 250,
        "max_area_px": 900,
        "hsv_ranges": {
            color: [{"lower": [max(0, hue - 2), 150, 100], "upper": [min(179, hue + 2), 255, 255]}]
            for color, hue in HUES.items()
        },
    }


def write_synthetic(path: Path) -> None:
    hsv = np.zeros((240, 360, 3), dtype=np.uint8)
    hsv[:, :, 2] = 30
    for index, color in enumerate(COLORS):
        row, column = divmod(index, 3)
        x, y = 35 + column * 110, 45 + row * 110
        hsv[y:y + 25, x:x + 25] = (HUES[color], 220, 210)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    self_encoded, payload = cv2.imencode(".png", bgr)
    assert self_encoded
    payload.tofile(path)


class OfflineAnalyzerTest(unittest.TestCase):
    def test_batch_discovery_and_analysis_use_all_six_colors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "中文图片"
            root.mkdir()
            image_path = root / "point-1.png"
            write_synthetic(image_path)
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            self.assertEqual(discover_images([root]), [image_path.resolve()])
            result = analyze_image(image_path, detector())
            self.assertTrue(result.summary["success"])
            self.assertEqual([report["color"] for report in result.summary["colors"]], list(COLORS))
            self.assertEqual(result.mask_bgr.shape, result.original_bgr.shape)
            self.assertEqual(result.annotated_bgr.shape, result.original_bgr.shape)
            self.assertEqual(result.returned_bgr.shape, result.original_bgr.shape)
            self.assertFalse(np.array_equal(result.returned_bgr, result.original_bgr))

    def test_annotated_evidence_is_replaced_with_raw_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "point-2.png"
            annotated = root / "point-2-annotated.png"
            write_synthetic(raw)
            write_synthetic(annotated)

            self.assertEqual(discover_images([annotated]), [raw.resolve()])
            self.assertEqual(discover_images([root]), [raw.resolve()])

    def test_candidate_is_separate_and_records_no_formal_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"; write_synthetic(image_path)
            output = root / "candidate.json"
            write_candidate(output, scene="trays", detector=detector(), source_images=[image_path])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "offline_detector_candidate")
            self.assertEqual(payload["scene"], "trays")
            self.assertFalse(payload["formal_camera_json_modified"])
            self.assertEqual(set(payload["detector"]["hsv_ranges"]), set(COLORS))

    def test_invalid_hsv_is_rejected_before_analysis(self) -> None:
        value = detector(); value["hsv_ranges"].pop(COLORS[-1])
        with self.assertRaisesRegex(OfflineAnalysisError, "完整包含"):
            normalize_detector_config(value)

    def test_best_effort_returns_contour_rejected_by_strict_filters(self) -> None:
        image_hsv = np.zeros((120, 120, 3), dtype=np.uint8)
        image_hsv[:, :, 2] = 20
        image_hsv[40:60, 45:65] = (HUES["红"], 220, 210)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        value = detector()
        value["roi"] = [0, 0, 120, 120]
        value["min_area_px"] = 1000
        value["max_area_px"] = 2000

        strict = analyze_color_candidates(image, "红", value)
        preview = analyze_color_candidates(image, "红", value, selection_policy="best_effort")

        self.assertEqual(strict["status"], "not_found")
        self.assertEqual(preview["status"], "success")
        self.assertIsNotNone(preview["selected"])
        self.assertIn("area_below_min", preview["warnings"])

    def test_best_effort_returns_largest_area(self) -> None:
        image_hsv = np.zeros((160, 200, 3), dtype=np.uint8)
        image_hsv[:, :, 2] = 20
        image_hsv[20:40, 20:40] = (HUES["红"], 220, 210)
        image_hsv[80:120, 100:140] = (HUES["红"], 220, 210)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        value = detector()
        value["roi"] = [0, 0, 200, 160]
        value["min_area_px"] = 1
        value["max_area_px"] = 5000

        report = analyze_color_candidates(image, "红", value, selection_policy="best_effort")

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["candidate_count"], 2)
        self.assertGreater(report["selected"]["area_px"], 1000)
        self.assertIn("multiple_candidates_best_selected", report["warnings"])

    def test_best_effort_prefers_larger_area_over_higher_confidence(self) -> None:
        image_hsv = np.zeros((180, 240, 3), dtype=np.uint8)
        image_hsv[:, :, 2] = 20
        # Smaller perfect square: confidence is close to 1.0.
        image_hsv[20:50, 20:50] = (HUES["绿"], 220, 210)
        # Larger rectangle: lower shape confidence but greater contour area.
        image_hsv[90:130, 100:170] = (HUES["绿"], 220, 210)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        value = detector()
        value["roi"] = [0, 0, 240, 180]
        value["min_area_px"] = 100
        value["max_area_px"] = 5000

        report = analyze_color_candidates(image, "绿", value, selection_policy="best_effort")

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["eligible_candidate_count"], 2)
        self.assertGreater(report["candidates"][1]["confidence"], report["selected"]["confidence"])
        self.assertGreater(report["selected"]["area_px"], report["candidates"][1]["area_px"])
        self.assertAlmostEqual(report["selected"]["center"][0], 134.5, delta=2.0)

    def test_best_effort_ignores_tiny_perfect_fragment_before_confidence_sort(self) -> None:
        image_hsv = np.zeros((160, 200, 3), dtype=np.uint8)
        image_hsv[:, :, 2] = 20
        image_hsv[10:13, 10:13] = (HUES["绿"], 220, 210)
        image_hsv[70:110, 90:130] = (HUES["绿"], 220, 210)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        value = detector()
        value["roi"] = [0, 0, 200, 160]
        value["min_area_px"] = 250
        value["max_area_px"] = 5000

        report = analyze_color_candidates(image, "绿", value, selection_policy="best_effort")

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["eligible_candidate_count"], 1)
        self.assertEqual(report["ignored_tiny_fragment_count"], 1)
        self.assertGreater(report["selected"]["area_px"], 1000)
        self.assertIn("tiny_fragments_ignored", report["warnings"])


if __name__ == "__main__":
    unittest.main()
