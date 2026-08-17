from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from app.robot_config_editor import COLORS, POINT_KEYS, RobotConfigInputError, load_reference_anchors, load_robot_editor_values, save_reference_anchor, save_robot_editor_values, save_single_joint_point


class RobotConfigEditorTest(unittest.TestCase):
    def _files(self, root: Path) -> tuple[Path, Path]:
        robot = root / "robot.json"
        motion = root / "motion.json"
        robot.write_text(json.dumps({
            "schema_version": 1, "identity": {"serial_number": "KEEP"},
            "active_tcp": {"name": "UNSET", "offset": "UNSET", "tolerance": 1e-8},
        }), encoding="utf-8")
        motion.write_text(json.dumps({
            "schema_version": 1, "points": {key: "UNSET" for key in POINT_KEYS},
            "direct_routes": {"a": True, "b": True}, "real_robot_verified": True,
            "limits": {"speed_fraction": 0.05},
        }), encoding="utf-8")
        return robot, motion

    def test_arcs_display_units_are_converted_and_other_fields_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            robot_path, motion_path = self._files(Path(temp))
            points = {key: [0, 30, 60, 90, -30, -60] for key in POINT_KEYS}
            save_robot_editor_values(
                robot_path, motion_path, tcp_name="round_tcp",
                tcp_values=[10, -20, 300, 0, 90, -45], tcp_units="mm_deg",
                point_values=points, joint_units="deg",
            )
            robot = json.loads(robot_path.read_text(encoding="utf-8"))
            motion = json.loads(motion_path.read_text(encoding="utf-8"))
            self.assertEqual(robot["identity"]["serial_number"], "KEEP")
            self.assertEqual(robot["active_tcp"]["name"], "round_tcp")
            self.assertAlmostEqual(robot["active_tcp"]["offset"][0], 0.01)
            self.assertAlmostEqual(robot["active_tcp"]["offset"][4], math.pi / 2)
            self.assertAlmostEqual(motion["points"]["task_card_photo"][1], math.pi / 6)
            self.assertNotIn("contact_z", motion)
            self.assertEqual(motion["direct_routes"], {"a": True, "b": True})
            self.assertFalse(motion["real_robot_verified"])
            name, offset, loaded_points = load_robot_editor_values(robot_path, motion_path)
            self.assertEqual(name, "round_tcp"); self.assertEqual(len(offset or []), 6)
            self.assertEqual(len(loaded_points["blocks_photo"] or []), 6)

    def test_incomplete_or_nonfinite_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            robot_path, motion_path = self._files(Path(temp))
            with self.assertRaises(RobotConfigInputError):
                save_robot_editor_values(
                    robot_path, motion_path, tcp_name="tcp", tcp_values=[0] * 5,
                    tcp_units="m_rad", point_values={key: [0] * 6 for key in POINT_KEYS},
                    joint_units="rad",
                )

    def test_single_point_capture_saves_without_other_fields_and_revokes_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _robot_path, motion_path = self._files(Path(temp))
            motion = json.loads(motion_path.read_text(encoding="utf-8"))
            motion["nine_point"] = {"blocks": {"automatic_verified": True}, "trays": {"automatic_verified": True}}
            motion["reference_anchors"] = {"blocks": [0.1] * 6, "trays": [0.2] * 6}
            motion_path.write_text(json.dumps(motion), encoding="utf-8")
            saved = save_single_joint_point(
                motion_path, point_key="blocks_photo", joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            )
            result = json.loads(motion_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            self.assertEqual(result["points"]["blocks_photo"], saved)
            self.assertEqual(result["points"]["task_card_photo"], "UNSET")
            self.assertFalse(result["real_robot_verified"])
            self.assertFalse(result["nine_point"]["blocks"]["automatic_verified"])
            self.assertTrue(result["nine_point"]["trays"]["automatic_verified"])
            self.assertEqual(result["reference_anchors"]["blocks"], {color: "UNSET" for color in COLORS})
            self.assertEqual(result["reference_anchors"]["trays"]["红"], [0.2] * 6)
            self.assertTrue(all(result["reference_anchors"]["trays"][color] == "UNSET" for color in COLORS[1:]))

    def test_reference_anchor_is_saved_as_complete_tcp_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _robot_path, motion_path = self._files(Path(temp))
            pose = [0.31, -0.22, 0.15, 3.14, 0.0, 0.2]
            self.assertEqual(save_reference_anchor(motion_path, scene="blocks", tcp_pose=pose), pose)
            anchors = load_reference_anchors(motion_path)
            self.assertEqual(anchors["blocks"]["红"], pose)
            self.assertTrue(all(anchors["blocks"][color] is None for color in COLORS[1:]))
            self.assertTrue(all(anchors["trays"][color] is None for color in COLORS))

    def test_legacy_red_anchors_are_preserved_while_saving_all_twelve(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _robot_path, motion_path = self._files(Path(temp))
            motion = json.loads(motion_path.read_text(encoding="utf-8"))
            old_block = [0.1] * 6; old_tray = [0.2] * 6
            motion["reference_anchors"] = {"blocks": old_block, "trays": old_tray}
            motion_path.write_text(json.dumps(motion), encoding="utf-8")
            loaded = load_reference_anchors(motion_path)
            self.assertEqual(loaded["blocks"]["红"], old_block)
            self.assertEqual(loaded["trays"]["红"], old_tray)
            for scene_index, scene in enumerate(("blocks", "trays")):
                for color_index, color in enumerate(COLORS):
                    pose = [0.3 + scene_index * 0.1, color_index / 100.0, 0.15, 3.14, 0.0, 0.1]
                    save_reference_anchor(motion_path, scene=scene, color=color, tcp_pose=pose)
            anchors = load_reference_anchors(motion_path)
            self.assertTrue(all(anchors[scene][color] is not None for scene in ("blocks", "trays") for color in COLORS))

    def test_reference_anchor_rejects_bad_color_length_and_nonfinite_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _robot_path, motion_path = self._files(Path(temp))
            with self.assertRaises(RobotConfigInputError):
                save_reference_anchor(motion_path, scene="blocks", color="白", tcp_pose=[0.1] * 6)
            with self.assertRaises(RobotConfigInputError):
                save_reference_anchor(motion_path, scene="blocks", color="红", tcp_pose=[0.1] * 5)
            with self.assertRaises(RobotConfigInputError):
                save_reference_anchor(motion_path, scene="blocks", color="红", tcp_pose=[0.1] * 5 + [math.nan])

if __name__ == "__main__":
    unittest.main()
