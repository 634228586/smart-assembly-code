from __future__ import annotations

import unittest

from app.calibration_worker import calibration_motion_limits


class CalibrationWorkerLimitsTest(unittest.TestCase):
    def motion(self) -> dict:
        return {
            "nine_point": {
                "speed_fraction": 0.5,
                "linear_acceleration_m_s2": 0.1,
                "linear_velocity_m_s": 0.03,
            },
            "limits": {
                "speed_fraction": 0.7,
                "maximum_authorized_speed_fraction": 0.7,
                "tcp_position_tolerance_m": 0.001,
            },
        }

    def test_builds_center_and_grid_limits_without_undefined_variable(self) -> None:
        center, grid = calibration_motion_limits(self.motion())
        self.assertEqual(center["speed_fraction"], 0.7)
        self.assertEqual(grid["speed_fraction"], 0.5)
        self.assertEqual(grid["tcp_position_tolerance_m"], 0.0005)

    def test_rejects_grid_speed_above_authorized_maximum(self) -> None:
        motion = self.motion()
        motion["limits"]["maximum_authorized_speed_fraction"] = 0.4
        with self.assertRaisesRegex(RuntimeError, "低于九点网格运动速度"):
            calibration_motion_limits(motion)


if __name__ == "__main__":
    unittest.main()
