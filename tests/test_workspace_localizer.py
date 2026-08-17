from __future__ import annotations

import unittest

import numpy as np

from vision.workspace_localizer import ApprovedCalibration, COLORS, build_detection_diagnostic, locate_colors


class WorkspaceLocalizerTest(unittest.TestCase):
    def test_diagnostic_hides_microscopic_reject_fragments_but_keeps_them_in_summary(self) -> None:
        color = COLORS[0]
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[70:131, 70:131] = (255, 255, 255)
        image[20:25, 20:25] = (255, 255, 255)
        profile = {
            "detector": {
                "approved": True, "roi": [0, 0, 200, 200],
                "confidence_min": 0.6, "min_area_px": 1000, "max_area_px": 10000,
                "hsv_ranges": {color: [{"lower": [0, 0, 1], "upper": [179, 255, 255]}]},
            }
        }

        marked, summary = build_detection_diagnostic(image, profile=profile, colors=(color,))

        candidates = summary["colors"][0]["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(any(candidate["selected"] for candidate in candidates))
        self.assertTrue(any(not candidate["accepted"] for candidate in candidates))
        # The small white fragment remains unchanged: no red REJECT box or label is drawn.
        self.assertTrue(np.all(marked[18:28, 18:28, 0] == marked[18:28, 18:28, 1]))

    def test_unapproved_detector_still_runs_real_uniqueness_detection(self) -> None:
        color = COLORS[0]
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[70:131, 70:131] = (255, 255, 255)
        profile = {
            "detector": {
                "approved": False,
                "roi": [0, 0, 200, 200],
                "confidence_min": 0.6,
                "min_area_px": 1000,
                "max_area_px": 10000,
                "hsv_ranges": {
                    color: [{"lower": [0, 0, 1], "upper": [179, 255, 255]}],
                },
            }
        }
        calibration = ApprovedCalibration(
            scene="blocks",
            calibration_id="test",
            camera_serial="test-camera",
            active_tcp="test-tcp",
            photo_point="blocks_photo",
            image_width=200,
            image_height=200,
            homography=np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
            reference_detections={color: {"pixel_u": 100.0, "pixel_v": 100.0, "r_image_deg": 0.0, "confidence": 1.0}},
        )

        result = locate_colors(
            image, profile=profile, calibration=calibration, requested_color=color
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["color"], color)
        self.assertGreaterEqual(result[0]["confidence"], 0.6)
        self.assertAlmostEqual(result[0]["dx_tool_m"], 0.0, delta=0.003)
        self.assertAlmostEqual(result[0]["dy_tool_m"], 0.0, delta=0.003)
        self.assertAlmostEqual(result[0]["delta_x_tool_m"], 0.0, delta=0.003)
        self.assertAlmostEqual(result[0]["delta_y_tool_m"], 0.0, delta=0.003)


if __name__ == "__main__":
    unittest.main()
