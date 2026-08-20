from __future__ import annotations

import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.robot_gateway import AuboRealGateway, MotionPermit, RobotGatewayError, RobotSnapshot


class FakeIo:
    def __init__(self) -> None: self.outputs = {}; self.tool_outputs = {}; self.tool_inputs = {0: True, 1: True}; self.input = True; self.writes = []; self.tool_writes = []
    def setStandardDigitalOutput(self, index, level): self.writes.append((index, level)); self.outputs[index] = level; return 0
    def getStandardDigitalOutput(self, index): return self.outputs.get(index, False)
    def getStandardDigitalInput(self, index): return self.input
    def isToolIoInput(self, index): return self.tool_inputs.get(index, True)
    def setToolIoInput(self, index, is_input): self.tool_inputs[index] = is_input; return 0
    def setToolDigitalOutput(self, index, level): self.tool_writes.append((index, level)); self.tool_outputs[index] = level; return 0
    def getToolDigitalOutput(self, index): return self.tool_outputs.get(index, False)


class FakeRobot:
    def __init__(self, io): self.io = io
    def getIoControl(self): return self.io


class FakeMotion:
    def __init__(self) -> None: self.speed_calls = []; self.joint_calls = []; self.line_calls = []
    def setSpeedFraction(self, value): self.speed_calls.append(value); return 0
    def moveJoint(self, *args): self.joint_calls.append(args); return 0
    def moveLine(self, *args): self.line_calls.append(args); return 0


class GatewayForTest(AuboRealGateway):
    def __init__(self) -> None:
        super().__init__({"host": "unused", "port": 1}, {
            "required_state": {"robot_mode": "Running", "safety_mode": "Normal", "exec_id": -1, "runtime_machine_running": True},
            "active_tcp": {"offset": [0, 0, 0, 0, 0, 0], "tolerance": 1e-8},
        })
        self.io = FakeIo(); self.motion = FakeMotion(); self.client = object(); self.robot_name = "robot"
    def _interfaces(self): return FakeRobot(self.io), object(), self.motion
    def start_runtime_for_maintenance(self, permit, fingerprint, timeout_s=5.0):
        permit.assert_valid(fingerprint)
        return False
    def assert_motion_gate(self, permit, fingerprint):
        permit.assert_valid(fingerprint)
        return RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6, (0,) * 6, -1, "RuntimeState.Running")
    def assert_io_gate(self, permit, fingerprint):
        permit.assert_valid(fingerprint)
        return RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6, (0,) * 6, -1, "RuntimeState.Stopped")


class RobotGatewaySafetyTest(unittest.TestCase):
    def test_snapshot_retries_transient_sdk_unicode_decode_error(self) -> None:
        gateway = AuboRealGateway({"host": "unused", "port": 1}, {})
        gateway.client = Mock()
        gateway.robot_name = "rob1"
        robot = Mock()
        state = Mock()
        motion = Mock()
        gateway._interfaces = Mock(side_effect=[UnicodeDecodeError("utf-8", b"\xd3", 0, 1, "bad"), (robot, state, motion)])
        state.getRobotModeType.return_value = "Running"
        state.getSafetyModeType.return_value = "Normal"
        state.getJointPositions.return_value = [0.0] * 6
        state.getTcpPose.return_value = [0.0] * 6
        robot.getRobotConfig.return_value.getTcpOffset.return_value = [0.0] * 6
        motion.getExecId.return_value = -1
        gateway.client.getRuntimeMachine.return_value.getStatus.return_value = "Running"

        snapshot = gateway.snapshot()

        self.assertEqual(snapshot.robot_name, "rob1")
        self.assertEqual(gateway._interfaces.call_count, 2)

    def permit(self): return MotionPermit("session", time.monotonic(), "fingerprint")

    def test_speed_fraction_cannot_exceed_approved_maximum(self) -> None:
        gateway = GatewayForTest()
        limits = {"speed_fraction": 0.2, "maximum_authorized_speed_fraction": 0.1, "joint_acceleration_rad_s2": 1.0, "joint_velocity_rad_s": 1.0}
        with self.assertRaises(RobotGatewayError):
            gateway.move_joint([0] * 6, limits, self.permit(), "fingerprint")
        self.assertEqual(gateway.motion.joint_calls, [])

    def test_maintenance_point_motion_requires_running_runtime(self) -> None:
        gateway = AuboRealGateway({"host": "unused", "port": 1}, {
            "required_state": {"robot_mode": "Running", "safety_mode": "Normal", "exec_id": -1, "runtime_machine_running": True},
            "active_tcp": {"offset": [0, 0, 0, 0, 0, 0], "tolerance": 1e-8},
        })
        gateway.snapshot = lambda: RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Stopped",
        )
        with self.assertRaisesRegex(RobotGatewayError, "RuntimeMachine未启动"):
            gateway.assert_maintenance_motion_gate(self.permit(), "fingerprint")
        with self.assertRaises(RobotGatewayError):
            gateway.assert_motion_gate(self.permit(), "fingerprint")

    def test_maintenance_runtime_start_does_not_run_program(self) -> None:
        gateway = AuboRealGateway({"host": "unused", "port": 1}, {
            "required_state": {"robot_mode": "Running", "safety_mode": "Normal", "exec_id": -1, "runtime_machine_running": True},
            "active_tcp": {"offset": [0, 0, 0, 0, 0, 0], "tolerance": 1e-8},
        })
        stopped = RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Stopped",
        )
        running = RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Running",
        )
        runtime = SimpleNamespace(start=Mock(return_value=0), runProgram=Mock())
        gateway.client = SimpleNamespace(getRuntimeMachine=lambda: runtime)
        gateway.robot_name = "robot"
        gateway.assert_maintenance_gate = Mock(return_value=stopped)
        gateway.snapshot = Mock(return_value=running)
        self.assertTrue(gateway.start_runtime_for_maintenance(self.permit(), "fingerprint"))
        runtime.start.assert_called_once_with()
        runtime.runProgram.assert_not_called()

    def test_maintenance_move_joint_automatically_starts_runtime(self) -> None:
        gateway = AuboRealGateway({"host": "unused", "port": 1}, {
            "required_state": {"robot_mode": "Running", "safety_mode": "Normal", "exec_id": -1, "runtime_machine_running": True},
            "active_tcp": {"offset": [0, 0, 0, 0, 0, 0], "tolerance": 1e-8},
        })
        stopped = RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Stopped",
        )
        running = RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Running",
        )
        runtime = SimpleNamespace(start=Mock(return_value=0), runProgram=Mock())
        motion = FakeMotion()
        gateway.client = SimpleNamespace(getRuntimeMachine=lambda: runtime)
        gateway.robot_name = "robot"
        gateway.snapshot = Mock(side_effect=[stopped, running, running])
        gateway._interfaces = Mock(return_value=(FakeRobot(FakeIo()), object(), motion))
        limits = {
            "speed_fraction": 0.1, "maximum_authorized_speed_fraction": 0.4,
            "joint_acceleration_rad_s2": 1.0, "joint_velocity_rad_s": 1.0,
        }

        self.assertEqual(
            gateway.move_joint_maintenance([0] * 6, limits, self.permit(), "fingerprint"),
            0,
        )
        runtime.start.assert_called_once_with()
        runtime.runProgram.assert_not_called()
        self.assertEqual(len(motion.joint_calls), 1)

    def test_wait_does_not_treat_delayed_exec_id_as_already_finished(self) -> None:
        class DelayedExecMotion(FakeMotion):
            def __init__(self) -> None:
                super().__init__(); self.exec_reads = 0
            def getExecId(self):
                self.exec_reads += 1
                return -1 if self.exec_reads == 1 else 153

        gateway = GatewayForTest(); gateway.motion = DelayedExecMotion()
        target = (0.4, 0.1, 0.5, 0.0, 0.0, 0.0)
        snapshots = iter([
            RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, target, (0,) * 6, -1, "RuntimeState.Running"),
            RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, target, (0,) * 6, 153, "RuntimeState.Running"),
            RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, target, (0,) * 6, -1, "RuntimeState.Running"),
            RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, target, (0,) * 6, -1, "RuntimeState.Running"),
            RobotSnapshot("robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, target, (0,) * 6, -1, "RuntimeState.Running"),
        ])
        gateway.snapshot = lambda: next(snapshots)

        gateway._wait_until_target(
            target, kind="tcp", position_tolerance=0.001,
            orientation_tolerance=0.01, timeout_s=2.0,
        )

        self.assertGreaterEqual(gateway.motion.exec_reads, 2)

    def test_wait_allows_short_pose_status_settle_after_queue_finishes(self) -> None:
        class RunningMotion(FakeMotion):
            def getExecId(self): return 153

        gateway = GatewayForTest(); gateway.motion = RunningMotion()
        target = (0.4, 0.1, 0.5, 0.0, 0.0, 0.0)
        snapshots = iter([
            RobotSnapshot(
                "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6,
                (0.4012, 0.1, 0.5, 0.0, 0.0, 0.0), (0,) * 6, -1,
                "RuntimeState.Running",
            ),
            RobotSnapshot(
                "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6,
                target, (0,) * 6, -1, "RuntimeState.Running",
            ),
            RobotSnapshot(
                "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6,
                target, (0,) * 6, -1, "RuntimeState.Running",
            ),
            RobotSnapshot(
                "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6,
                target, (0,) * 6, -1, "RuntimeState.Running",
            ),
        ])
        gateway.snapshot = lambda: next(snapshots)

        gateway._wait_until_target(
            target, kind="tcp", position_tolerance=0.001,
            orientation_tolerance=0.01, timeout_s=2.0,
        )

    def test_final_position_error_reports_target_actual_and_residual(self) -> None:
        class RunningMotion(FakeMotion):
            def getExecId(self): return 153

        gateway = GatewayForTest(); gateway.motion = RunningMotion()
        target = (0.4, 0.1, 0.5, 0.0, 0.0, 0.0)
        gateway.snapshot = lambda: RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6,
            (0.402, 0.1, 0.5, 0.0, 0.0, 0.0), (0,) * 6, -1,
            "RuntimeState.Running",
        )

        with patch("app.robot_gateway.FINAL_TARGET_SETTLE_S", 0.0):
            with self.assertRaisesRegex(
                RobotGatewayError,
                "最终位置误差超限.*目标TCP=.*实际TCP=.*Δ位置",
            ):
                gateway._wait_until_target(
                    target, kind="tcp", position_tolerance=0.001,
                    orientation_tolerance=0.01, timeout_s=2.0,
                )

    def test_orientation_comparison_accepts_equivalent_angles_across_pi_seam(self) -> None:
        target = (0.4, 0.1, 0.5, math.pi - 0.001, 0.0, 0.0)
        actual = (0.4, 0.1, 0.5, -math.pi + 0.001, 0.0, 0.0)

        self.assertTrue(AuboRealGateway._target_reached(
            actual, target, kind="tcp", position_tolerance=0.001,
            orientation_tolerance=0.01,
        ))

    def test_runtime_start_failure_prevents_motion(self) -> None:
        gateway = AuboRealGateway({"host": "unused", "port": 1}, {
            "required_state": {"robot_mode": "Running", "safety_mode": "Normal", "exec_id": -1, "runtime_machine_running": True},
            "active_tcp": {"offset": [0, 0, 0, 0, 0, 0], "tolerance": 1e-8},
        })
        stopped = RobotSnapshot(
            "robot", "RobotMode.Running", "SafetyMode.Normal", (0,) * 6, (0,) * 6,
            (0,) * 6, -1, "RuntimeState.Stopped",
        )
        runtime = SimpleNamespace(start=Mock(return_value=13), runProgram=Mock())
        motion = FakeMotion()
        gateway.client = SimpleNamespace(getRuntimeMachine=lambda: runtime)
        gateway.robot_name = "robot"
        gateway.snapshot = Mock(return_value=stopped)
        gateway._interfaces = Mock(return_value=(FakeRobot(FakeIo()), object(), motion))
        limits = {
            "speed_fraction": 0.1, "maximum_authorized_speed_fraction": 0.4,
            "joint_acceleration_rad_s2": 1.0, "joint_velocity_rad_s": 1.0,
        }

        with self.assertRaisesRegex(RobotGatewayError, "RuntimeMachine.start失败"):
            gateway.move_joint_maintenance([0] * 6, limits, self.permit(), "fingerprint")
        self.assertEqual(motion.joint_calls, [])
        runtime.runProgram.assert_not_called()

    def test_maintenance_line_uses_maintenance_gate(self) -> None:
        gateway = GatewayForTest()
        gateway.assert_maintenance_motion_gate = gateway.assert_io_gate
        gateway.assert_motion_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("maintenance line must not require runtime"))
        limits = {
            "speed_fraction": 0.4, "maximum_authorized_speed_fraction": 0.4,
            "linear_acceleration_m_s2": 0.1, "linear_velocity_m_s": 0.03,
        }
        self.assertEqual(gateway.move_line_maintenance([0] * 6, limits, self.permit(), "fingerprint"), 0)
        self.assertEqual(len(gateway.motion.line_calls), 1)
        self.assertEqual(gateway.motion.speed_calls, [0.4])

    def test_stop_before_io_write_keeps_output_untouched(self) -> None:
        gateway = GatewayForTest()
        with self.assertRaises(RobotGatewayError):
            gateway.set_suction(True, {
                "output_type": "tool_digital_output", "output_index": 0, "enable_output_index": 1, "enable_level": True,
                "suction_level": True, "release_level": False,
                "feedback_mode": "none", "confirmation_timeout_s": 0.1,
            }, self.permit(), "fingerprint", should_stop=lambda: True)
        self.assertEqual(gateway.io.writes, [])

    def test_optional_real_vacuum_input_is_confirmed(self) -> None:
        gateway = GatewayForTest()
        result = gateway.set_suction(True, {
            "output_type": "tool_digital_output", "output_index": 0, "enable_output_index": 1, "enable_level": True,
            "suction_level": True, "release_level": False,
            "feedback_mode": "standard_digital_input", "feedback_index": 2,
            "feedback_active_level": True, "confirmation_timeout_s": 0.1,
        }, self.permit(), "fingerprint")
        self.assertTrue(result)
        self.assertEqual(gateway.io.tool_writes, [(1, True), (0, True)])

    def test_aperture_toggle_reads_do0_and_confirms_each_transition(self) -> None:
        gateway = GatewayForTest()
        enabled = gateway.toggle_standard_digital_output(
            index=0, on_level=True, off_level=False,
            permit=self.permit(), fingerprint="fingerprint", label="光圈 DO",
        )
        self.assertTrue(enabled)
        disabled = gateway.toggle_standard_digital_output(
            index=0, on_level=True, off_level=False,
            permit=self.permit(), fingerprint="fingerprint", label="光圈 DO",
        )
        self.assertFalse(disabled)
        self.assertEqual(gateway.io.writes, [(0, True), (0, False)])

    def test_standard_do_waits_for_delayed_controller_readback(self) -> None:
        class DelayedReadIo(FakeIo):
            def __init__(self) -> None:
                super().__init__(); self.visible = False; self.reads_after_write = 0
            def setStandardDigitalOutput(self, index, level):
                self.writes.append((index, level)); self.outputs[index] = level; self.reads_after_write = 0; return 0
            def getStandardDigitalOutput(self, index):
                self.reads_after_write += 1
                if self.reads_after_write >= 2:
                    self.visible = self.outputs.get(index, False)
                return self.visible

        gateway = GatewayForTest(); gateway.io = DelayedReadIo()
        result = gateway.set_standard_digital_output(
            index=0, level=True, permit=self.permit(), fingerprint="fingerprint",
            label="光圈 DO", confirmation_timeout_s=0.2,
        )
        self.assertTrue(result)
        self.assertGreaterEqual(gateway.io.reads_after_write, 2)

    def test_confirmed_suction_mapping_uses_tool0_and_keeps_tool1_enabled(self) -> None:
        gateway = GatewayForTest()
        config = {
            "output_type": "tool_digital_output", "output_index": 0,
            "enable_output_index": 1, "enable_level": True,
            "suction_level": True, "release_level": False,
            "feedback_mode": "none", "confirmation_timeout_s": 0.1,
        }
        gateway.set_suction(True, config, self.permit(), "fingerprint")
        gateway.set_suction(False, config, self.permit(), "fingerprint")
        self.assertEqual(gateway.io.tool_writes, [(1, True), (0, True), (1, True), (0, False)])
        self.assertTrue(gateway.io.tool_outputs[1])
        self.assertFalse(gateway.io.tool_outputs[0])

    def test_manual_do_does_not_require_motion_gate(self) -> None:
        gateway = GatewayForTest()
        def forbidden_motion_gate(*_args, **_kwargs):
            raise AssertionError("手动DO不应调用运动门控")
        gateway.assert_motion_gate = forbidden_motion_gate
        result = gateway.set_standard_digital_output(
            index=0, level=True, permit=self.permit(), fingerprint="fingerprint", label="测试DO",
        )
        self.assertTrue(result)
        self.assertEqual(gateway.io.writes, [(0, True)])


if __name__ == "__main__":
    unittest.main()
