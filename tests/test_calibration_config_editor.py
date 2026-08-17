from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.calibration_config_editor import load_calibration_settings, save_calibration_settings


class CalibrationConfigEditorTest(unittest.TestCase):
    def test_fixed_fifty_percent_and_step_change_revokes_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "motion.json"
            path.write_text(json.dumps({"schema_version": 1, "nine_point": {
                "speed_fraction": 0.15, "linear_acceleration_m_s2": "UNSET", "linear_velocity_m_s": "UNSET", "settle_s": 1,
                "blocks": {"step_x_mm": 20, "step_y_mm": 15, "target_color": "红", "automatic_verified": True},
                "trays": {"step_x_mm": 20, "step_y_mm": 15, "target_color": "红", "automatic_verified": True},
            }, "keep": 7}), encoding="utf-8")
            changed = save_calibration_settings(path, linear_acceleration_m_s2=0.2, linear_velocity_m_s=0.1, settle_s=1,
                scenes={"blocks": {"step_x_mm": 25, "step_y_mm": 15, "target_color": "红"}, "trays": {"step_x_mm": 20, "step_y_mm": 15, "target_color": "红"}})
            self.assertEqual(changed, {"blocks"})
            value = load_calibration_settings(path)
            self.assertEqual(value["speed_fraction"], 0.5); self.assertFalse(value["blocks"]["automatic_verified"]); self.assertTrue(value["trays"]["automatic_verified"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["keep"], 7)


if __name__ == "__main__":
    unittest.main()
