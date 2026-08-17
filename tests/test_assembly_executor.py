from __future__ import annotations

import unittest

from app.assembly import AssemblyError, AssemblyExecutor
from app.vision_client import VisionClientError, VisionTargetNotFoundError


COLORS = ["红", "橙", "黄", "绿", "蓝", "紫"]


def colour_anchors(block_z=0.1, tray_z=0.12):
    return {
        "blocks": {color: [0.3 + index / 100.0, 0.2, block_z, 3.14, 0, 0] for index, color in enumerate(COLORS)},
        "trays": {color: [0.5 + index / 100.0, 0.4, tray_z, 3.14, 0, 0] for index, color in enumerate(COLORS)},
    }


class FakeRobot:
    def __init__(self) -> None:
        self.joints = []; self.lines = []; self.io = []; self.transforms = []
        self.pose = (0.4, 0.1, 0.5, 3.14, 0.0, 0.0)
    def current_tcp_pose(self): return self.pose
    def pose_trans(self, base_pose, tool_delta):
        self.transforms.append((base_pose, tool_delta))
        return tuple(a + b for a, b in zip(base_pose, tool_delta))
    def move_joint(self, target): self.joints.append(tuple(target))
    def move_line(self, target): self.lines.append(tuple(target))
    def set_suction(self, enabled): self.io.append(enabled)
    def request_stop(self): pass


class FakeVision:
    def __init__(self) -> None: self.blocks = 0; self.trays = 0; self.request_ids = []
    def locate_block(self, **kwargs):
        self.blocks += 1; self.request_ids.append(kwargs["request_id"])
        return {"dx_tool_m": 0.001, "dy_tool_m": -0.001, "r_image_rad": 0.1, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.01}
    def locate_trays(self, **kwargs):
        self.trays += 1; self.request_ids.append(kwargs["request_id"])
        return {color: {"dx_tool_m": index / 1000, "dy_tool_m": 0.0, "r_image_rad": 0.2, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.02} for index, color in enumerate(COLORS)}


class RetryVision(FakeVision):
    def locate_block(self, **kwargs):
        self.blocks += 1; self.request_ids.append(kwargs["request_id"])
        if self.blocks == 1: raise VisionTargetNotFoundError("target not found")
        return {"dx_tool_m": 0.001, "dy_tool_m": -0.001, "r_image_rad": 0.1, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.01}


class MissingBlockVision(FakeVision):
    def locate_block(self, **kwargs):
        self.blocks += 1; self.request_ids.append(kwargs["request_id"])
        if kwargs["color"] == "红":
            raise VisionTargetNotFoundError("target not found")
        return {"dx_tool_m": 0.001, "dy_tool_m": -0.001, "r_image_rad": 0.1, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.01}


class MissingTrayVision(FakeVision):
    def locate_trays(self, **kwargs):
        result = super().locate_trays(**kwargs)
        result.pop("橙")
        return result


class CommunicationFailureVision(FakeVision):
    def locate_block(self, **kwargs):
        self.blocks += 1
        raise VisionClientError("connection lost")


class AssemblyExecutorTest(unittest.TestCase):
    def sequence(self):
        return [{"order": i + 1, "block_color": color, "tray_color": COLORS[(i + 1) % 6]} for i, color in enumerate(COLORS)]

    def run_executor(self, vision=None):
        robot = FakeRobot(); vision = vision or FakeVision(); events = []
        executor = AssemblyExecutor(robot, vision, session_id="session-real", points={
            "blocks_photo": [1, 2, 3, 4, 5, 6], "trays_photo": [6, 5, 4, 3, 2, 1],
            "task_card_photo": [0, 1, 2, 3, 4, 5],
            "competition_standby": [9, 8, 7, 6, 5, 4],
        }, reference_anchors=colour_anchors(), progress=events.append)
        result = executor.run(self.sequence())
        return robot, vision, events, result

    def test_six_groups_and_return_order(self) -> None:
        robot, vision, events, result = self.run_executor()
        self.assertEqual(len(robot.joints), 20)
        self.assertEqual(len(robot.lines), 36)
        self.assertEqual(robot.io, [True, False] * 6)
        self.assertEqual(len(robot.transforms), 6)
        self.assertEqual(vision.blocks, 6)
        self.assertEqual(vision.trays, 1)
        self.assertEqual(result.completed_cycles, 6)
        self.assertEqual(result.processed_cycles, 6)
        self.assertEqual(result.skipped_cycles, 0)
        self.assertTrue(result.returned_to_task_card)
        self.assertFalse(result.suction_may_be_on)
        self.assertEqual(events[-1]["phase"], "return_task_card")
        self.assertTrue(any(event["phase"] == "block_visual_offset" and "局部变化" in event["message"] for event in events))
        self.assertTrue(any(event["phase"] == "tray_visual_offset" and "局部变化" in event["message"] for event in events))
        self.assertEqual(robot.transforms[0][0], (0.51, 0.4, 0.5, 3.14, 0.0, 0.0))
        self.assertEqual(robot.transforms[0][1][:2], (0.0, 0.0))
        self.assertEqual(robot.lines[1][2], 0.1)
        self.assertEqual(robot.lines[4][2], 0.12)

    def test_one_stationary_retry(self) -> None:
        vision = RetryVision()
        _, vision, events, _ = self.run_executor(vision)
        self.assertEqual(vision.blocks, 7)
        self.assertEqual(vision.request_ids[1:3], ["session-real-B1-A1", "session-real-B1-A2"])
        self.assertTrue(any(event["phase"] == "block_capture_retry" for event in events))

    def test_missing_block_is_retried_then_group_is_skipped(self) -> None:
        robot, vision, events, result = self.run_executor(MissingBlockVision())
        self.assertEqual(vision.blocks, 7)
        self.assertEqual(result.completed_cycles, 5)
        self.assertEqual(result.skipped_cycles, 1)
        self.assertEqual(result.skipped[0]["reason"], "block_not_found")
        self.assertEqual(robot.io, [True, False] * 5)
        self.assertTrue(any(event["phase"] == "cycle_skipped" for event in events))

    def test_missing_tray_is_retried_and_skipped_before_pick(self) -> None:
        robot, vision, _events, result = self.run_executor(MissingTrayVision())
        self.assertEqual(vision.trays, 2)
        self.assertEqual(vision.blocks, 5)
        self.assertEqual(result.completed_cycles, 5)
        self.assertEqual(result.skipped_cycles, 1)
        self.assertEqual(result.skipped[0]["reason"], "tray_not_found")
        self.assertEqual(robot.io, [True, False] * 5)

    def test_communication_failure_is_not_retried_or_skipped(self) -> None:
        vision = CommunicationFailureVision()
        with self.assertRaises(VisionClientError):
            self.run_executor(vision)
        self.assertEqual(vision.blocks, 1)

    def test_single_group_bypasses_task_card_and_returns_standby(self) -> None:
        robot = FakeRobot(); vision = FakeVision(); events = []
        executor = AssemblyExecutor(robot, vision, session_id="direct-one", points={
            "blocks_photo": [1, 2, 3, 4, 5, 6], "trays_photo": [6, 5, 4, 3, 2, 1],
            "task_card_photo": [0, 1, 2, 3, 4, 5], "competition_standby": [9, 8, 7, 6, 5, 4],
        }, reference_anchors=colour_anchors(tray_z=0.1), progress=events.append)
        result = executor.run_single(block_color="红", tray_color="绿")
        self.assertEqual(result.completed_cycles, 1)
        self.assertFalse(result.returned_to_task_card)
        self.assertEqual(vision.blocks, 1)
        self.assertEqual(vision.trays, 1)
        self.assertEqual(robot.io, [True, False])
        self.assertEqual(robot.joints[-1], (9, 8, 7, 6, 5, 4))
        self.assertEqual(events[-1]["phase"], "direct_return_standby")

    def test_missing_selected_colour_anchor_rejects_before_any_motion_or_suction(self) -> None:
        robot = FakeRobot(); anchors = colour_anchors(); anchors["trays"]["绿"] = "UNSET"
        executor = AssemblyExecutor(robot, FakeVision(), session_id="guard", points={
            "blocks_photo": [1] * 6, "trays_photo": [2] * 6, "task_card_photo": [3] * 6,
            "competition_standby": [4] * 6,
        }, reference_anchors=anchors)
        with self.assertRaisesRegex(AssemblyError, "trays/.*绿"):
            executor.run_single(block_color="红", tray_color="绿")
        self.assertEqual(robot.joints, []); self.assertEqual(robot.lines, []); self.assertEqual(robot.io, [])

    def test_uses_selected_colour_anchor_and_same_colour_delta_only(self) -> None:
        class DeltaVision(FakeVision):
            def locate_block(self, **kwargs):
                return {"dx_tool_m": 0.9, "dy_tool_m": 0.8, "delta_x_tool_m": 0.002, "delta_y_tool_m": -0.003, "delta_r_rad": 0.01}
            def locate_trays(self, **kwargs):
                return {"紫": {"dx_tool_m": 0.7, "dy_tool_m": 0.6, "delta_x_tool_m": -0.004, "delta_y_tool_m": 0.005, "delta_r_rad": 0.02}}
        robot = FakeRobot(); executor = AssemblyExecutor(robot, DeltaVision(), session_id="delta", points={
            "blocks_photo": [1] * 6, "trays_photo": [2] * 6, "task_card_photo": [3] * 6,
            "competition_standby": [4] * 6,
        }, reference_anchors=colour_anchors())
        executor.run_single(block_color="蓝", tray_color="紫")
        self.assertAlmostEqual(robot.lines[0][0], 0.338)
        self.assertAlmostEqual(robot.lines[0][1], 0.203)
        self.assertAlmostEqual(robot.transforms[0][0][0], 0.554)
        self.assertAlmostEqual(robot.transforms[0][0][1], 0.395)
        self.assertEqual(robot.transforms[0][1][:2], (0, 0))
        self.assertAlmostEqual(robot.transforms[0][1][5], -0.01)


if __name__ == "__main__":
    unittest.main()
