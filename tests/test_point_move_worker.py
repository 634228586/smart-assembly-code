from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.point_capture_worker import RobotPointMoveWorker


class RobotPointMoveWorkerTest(unittest.TestCase):
    def test_moves_to_saved_point_with_real_gateway_limits_and_waits(self) -> None:
        target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        limits = {
            "speed_fraction": 0.05, "maximum_authorized_speed_fraction": 0.15,
            "joint_acceleration_rad_s2": 0.3, "joint_velocity_rad_s": 0.15,
        }
        configs = {
            "endpoints": {"robot_rpc": {"host": "192.168.34.1", "port": 30004}},
            "robot": {"identity": {}},
            "motion": {
                "points": {"blocks_photo": target}, "limits": limits,
                "maintenance_point": {"speed_fraction": 0.7, "maximum_authorized_speed_fraction": 0.7},
            },
        }
        snapshot = SimpleNamespace(
            robot_name="rob1",
            joint_positions=tuple(target),
        )
        gateway = MagicMock()
        gateway.snapshot.return_value = snapshot
        finished: list[object] = []
        failed: list[str] = []
        worker = RobotPointMoveWorker("blocks_photo")
        worker.finished.connect(finished.append)
        worker.failed.connect(failed.append)

        with (
            patch("app.point_capture_worker.load_all", return_value=configs),
            patch("app.point_capture_worker.current_runtime_fingerprint", return_value="fingerprint"),
            patch("app.point_capture_worker.AuboRealGateway", return_value=gateway),
        ):
            worker.run()

        self.assertEqual(failed, [])
        self.assertEqual(len(finished), 1)
        gateway.connect_readonly.assert_called_once_with()
        gateway.move_joint_maintenance_and_wait.assert_called_once()
        gateway.move_joint_and_wait.assert_not_called()
        args, kwargs = gateway.move_joint_maintenance_and_wait.call_args
        self.assertEqual(args[0], target)
        self.assertIsNot(args[1], limits)
        self.assertEqual(args[1]["speed_fraction"], 0.7)
        self.assertEqual(args[1]["maximum_authorized_speed_fraction"], 0.7)
        self.assertEqual(args[3], "fingerprint")
        self.assertFalse(kwargs["should_stop"]())
        gateway.disconnect.assert_called_once_with()
        self.assertEqual(finished[0]["point_key"], "blocks_photo")
        self.assertEqual(finished[0]["speed_fraction"], 0.7)


if __name__ == "__main__":
    unittest.main()
