from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.coordinator import CompetitionCoordinator
from app.session import CompetitionSession, SessionState
from vision.contracts import CapturedFrame


COLORS = ["红", "橙", "黄", "绿", "蓝", "紫"]


class FakeRobot:
    def __init__(self) -> None:
        self.joints = []; self.lines = []; self.io = []; self.stop_requested = False
        self.pose = (0.4, 0.1, 0.5, 3.14, 0.0, 0.0)
    def current_tcp_pose(self): return self.pose
    def pose_trans(self, base_pose, delta): return tuple(a + b for a, b in zip(base_pose, delta))
    def move_joint(self, target): self.joints.append(tuple(target))
    def move_line(self, target): self.lines.append(tuple(target))
    def set_suction(self, enabled): self.io.append(enabled)
    def request_stop(self): self.stop_requested = True


class FakeUnifiedVision:
    def __init__(self) -> None: self.capture_count = 0; self.block_count = 0; self.tray_count = 0
    def capture_task_card(self, *, request_id, session_id, session_dir):
        self.capture_count += 1
        image = session_dir / "task_card" / f"card-{self.capture_count}.png"
        image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(b"fresh-frame")
        return CapturedFrame(request_id, "CAM-001", "task_card", image, datetime.now(timezone.utc).isoformat(), True)
    def locate_block(self, **kwargs):
        self.block_count += 1
        return {"dx_tool_m": 0.001, "dy_tool_m": -0.001, "r_image_rad": 0.1, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.01}
    def locate_trays(self, **kwargs):
        self.tray_count += 1
        return {color: {"dx_tool_m": index / 1000, "dy_tool_m": 0.0, "r_image_rad": 0.2, "delta_x_tool_m": 0.0, "delta_y_tool_m": 0.0, "delta_r_rad": 0.02} for index, color in enumerate(COLORS)}


class CompetitionCoordinatorTest(unittest.TestCase):
    def test_two_cards_run_in_one_session_and_tts_failure_does_not_block_task2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); session_dir = root / "session-1"
            session = CompetitionSession(); session.authorize(True)
            robot = FakeRobot(); vision = FakeUnifiedVision(); events = []; calls = {"recognizer": 0, "speaker": 0}

            def recognizer(_path):
                calls["recognizer"] += 1
                if calls["recognizer"] == 1:
                    return {"success": True, "task_type": "task_1", "confidence": 0.99, "scene_description": "装配现场正常"}
                return {
                    "success": True, "task_type": "task_2", "confidence": 0.99,
                    "sequence": [{"order": i + 1, "block_color": color, "tray_color": COLORS[(i + 1) % 6]} for i, color in enumerate(COLORS)],
                }

            def speaker(_text):
                calls["speaker"] += 1
                if calls["speaker"] == 2:
                    raise RuntimeError("tts unavailable")

            coordinator = CompetitionCoordinator(
                session=session, robot=robot, vision=vision, recognizer=recognizer,
                listener=lambda wakeup_required: "请", speaker=speaker,
                session_id="session-1", session_dir=session_dir, camera_serial="CAM-001",
                config_fingerprint="approved", points={
                    "task_card_photo": [0, 1, 2, 3, 4, 5],
                    "blocks_photo": [1, 2, 3, 4, 5, 6],
                    "trays_photo": [6, 5, 4, 3, 2, 1],
                }, reference_anchors={
                    "blocks": {color: [0.3, 0.2, 0.1, 3.14, 0, 0] for color in COLORS},
                    "trays": {color: [0.5, 0.4, 0.1, 3.14, 0, 0] for color in COLORS},
                }, progress=events.append,
            )
            coordinator.run()

            self.assertEqual(session.state, SessionState.COMPLETE)
            self.assertEqual(session.completed_cycles, 6)
            self.assertEqual(vision.capture_count, 2)
            self.assertEqual(vision.block_count, 6)
            self.assertEqual(vision.tray_count, 1)
            self.assertEqual(robot.io, [True, False] * 6)
            self.assertTrue(any(event["phase"] == "tts_warning" for event in events))
            self.assertEqual(robot.joints[-1], (0, 1, 2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
