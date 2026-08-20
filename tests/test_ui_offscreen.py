from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.preflight import competition_ready
from app.ui import CompetitionWindow


class UiOffscreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_authorization_is_disabled_with_unset_hardware(self) -> None:
        window = CompetitionWindow()
        try:
            self.assertFalse(window.authorize_button.isEnabled())
            self.assertIn("DO0", window.aperture_toggle_button.text())
            self.assertIn("TOOL_IO[1]=1", window.suction_on_button.text())
            self.assertIn("TOOL_IO[0]=0", window.suction_on_button.text())
            self.assertIn("TOOL_IO[0]=1", window.suction_off_button.text())
            self.assertIn("不假定", window.io_status.text())
            self.assertFalse(window.start_competition_button.isEnabled())
            self.assertFalse(window.stop_competition_button.isEnabled())
            self.assertFalse(window.text_mode_button.isChecked())
            self.assertFalse(window.text_input_edit.isEnabled())
            self.assertFalse(window.send_text_button.isEnabled())
            self.assertIn("发送给大模型测试", window.task_model_test_button.text())
            self.assertTrue(window.task_model_raw_text.isReadOnly())
            self.assertTrue(window.task_model_formal_text.isReadOnly())
            window.text_mode_button.click()
            self.app.processEvents()
            self.assertTrue(window.text_mode_button.isChecked())
            self.assertTrue(window.text_input_edit.isEnabled())
            self.assertFalse(window.send_text_button.isEnabled())
            self.assertIn("文字控制", window.text_control_status.text())
            expected_preflight_text = "全部通过" if competition_ready(window.checks) else "禁止比赛"
            self.assertIn(expected_preflight_text, window.preflight_status.text())
            self.assertGreater(window.preflight_table.rowCount(), 10)
            self.assertGreaterEqual(window.preflight_table.columnWidth(2), 300)
            self.assertIn("这项在检查什么：", window.preflight_detail.toPlainText())
            self.assertIn("当前发现：", window.preflight_detail.toPlainText())
            self.assertIn("去哪里处理：", window.preflight_detail.toPlainText())
            self.assertEqual(window.points_table.rowCount(), 4)
            self.assertEqual(window.points_table.columnCount(), 8)
            self.assertEqual(set(window.move_point_buttons), {
                "competition_standby", "task_card_photo", "blocks_photo", "trays_photo"
            })
            self.assertTrue(all("移动到此点" in button.text() for button in window.move_point_buttons.values()))
            self.assertIn("读取当前机械臂关节角", window.capture_point_button.text())
            self.assertFalse(hasattr(window, "contact_table"))
            self.assertFalse(hasattr(window, "capture_pick_z_button"))
            self.assertFalse(hasattr(window, "capture_place_z_button"))
            self.assertEqual(len(window.reference_anchor_buttons), 12)
            self.assertIn("Block/红色抓取基准", window.capture_blocks_reference_button.text())
            self.assertIn("Tray/红色放置基准", window.capture_trays_reference_button.text())
            self.assertFalse(hasattr(window, "tcp_name_edit"))
            self.assertFalse(hasattr(window, "tcp_table"))
            self.assertEqual(window.camera_profiles_table.rowCount(), 3)
            self.assertEqual(window.camera_profiles_table.columnCount(), 10)
            self.assertEqual(window.camera_serial_edit.text(), "DA5723714")
            self.assertIn("不回读", window.validate_profiles_button.text())
            self.assertFalse(window.read_mvs_parameters_button.isVisible())
            self.assertTrue(window.detector_editor_group.isHidden())
            self.assertIn("camera.json", window.detector_workflow_note.text())
            self.assertIn("Block/Tray 拍照取图", window.detector_workflow_note.text())
            self.assertIn("尚未拍摄", window.detector_visual_status.text())
            self.assertGreaterEqual(window.detector_visual_image.minimumHeight(), 200)
            self.assertEqual(window.manual_block_capture_button.text(), "Block 拍照取图")
            self.assertEqual(window.manual_tray_capture_button.text(), "Tray 拍照取图")
            self.assertIn("拍照并用当前参数识别", window.manual_block_recognize_button.text())
            self.assertIn("拍照并用当前参数识别", window.manual_tray_recognize_button.text())
            self.assertIn("尚未拍摄", window.monitor_visual_status.text())
            self.assertEqual(window.tabs.tabText(3), "单组抓放调试")
            self.assertEqual(window.direct_block_color_combo.count(), 6)
            self.assertEqual(window.direct_tray_color_combo.count(), 6)
            self.assertFalse(window.start_direct_assembly_button.isEnabled())
            window.direct_safety_check.setChecked(True)
            self.app.processEvents()
            self.assertTrue(window.start_direct_assembly_button.isEnabled())
            self.assertFalse(window.stop_direct_assembly_button.isEnabled())
            self.assertEqual(window.calibration_grid_table.rowCount(), 9)
            self.assertEqual(window.calibration_points_table.rowCount(), 9)
            window.calibration_scene_combo.setCurrentIndex(window.calibration_scene_combo.findData("blocks"))
            window.calibration_step_edits["blocks"][0].setText("12")
            window.calibration_step_edits["blocks"][1].setText("11")
            self.app.processEvents()
            self.assertEqual(window.calibration_grid_table.item(0, 1).text(), "-12.0")
            self.assertEqual(window.calibration_grid_table.item(0, 2).text(), "-11.0")
            self.assertEqual(window.calibration_grid_table.item(0, 3).text(), "12.0")
            self.assertEqual(window.calibration_grid_table.item(0, 4).text(), "11.0")
            self.assertFalse(window.calibration_grid_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
            self.assertFalse(window.accept_calibration_button.isEnabled())
            self.assertTrue(window.accept_calibration_button.isHidden())
            self.assertTrue(window.calibration_automatic_check.isChecked())
            self.assertTrue(all(label.text() == "一次连续完成" for label in window.calibration_auto_labels.values()))
            self.assertIn("跳过", window.calibration_validation_label.text())
            self.assertTrue(all(button.isHidden() for button in window.calibration_validation_buttons.values()))
            self.assertIn("只读检查机械臂", window.robot_readiness_button.text())
            self.assertIn("未检查", window.robot_readiness_status.text())
            self.assertFalse(window.direct_paths_check.isChecked())
            self.assertIn("不提供碰撞规划", window.direct_paths_check.text())
            self.assertFalse(window.authorize_button.isEnabled())
            self.assertIn("开始识别", window.voice_listen_button.text())
            self.assertTrue(window.voice_recognized_text.isReadOnly())
            self.assertIn("语音交互测试成功", window.voice_tts_edit.text())
            self.assertIn("尚未测试", window.voice_status.text())
        finally:
            window.close()

    def test_approving_nine_point_preserves_all_existing_colour_anchors(self) -> None:
        window = CompetitionWindow()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_dir = root / "config"
                calibration_dir = root / "calibration"
                evidence_dir = root / "evidence"
                config_dir.mkdir()
                block_pose = [0.31, 0.22, 0.18, 3.14, 0.0, 0.0]
                tray_pose = [0.51, 0.42, 0.18, 3.14, 0.0, 0.0]
                motion_path = config_dir / "motion.json"
                motion_path.write_text(json.dumps({
                    "schema_version": 1,
                    "reference_anchors": {"blocks": block_pose, "trays": tray_pose},
                }), encoding="utf-8")
                window.calibration_candidate = {
                    "scene": "blocks",
                    "candidate_path": str(root / "candidate.json"),
                }
                approved = {"calibration_id": "N9_BLOCKS_TEST", "rms_error_mm": 0.1, "max_error_mm": 0.2}
                with (
                    patch("app.ui.REAL_CONFIG_DIR", config_dir),
                    patch("app.ui.REAL_CALIBRATION_DIR", calibration_dir),
                    patch("app.ui.EVIDENCE_DIR", evidence_dir),
                    patch("app.ui.approve_candidate_without_direction_validation", return_value=approved),
                    patch("app.ui.mark_automatic_verified"),
                    patch.object(window, "_reload_calibration_settings"),
                    patch.object(window, "_reload_robot_config"),
                    patch.object(window, "run_preflight"),
                ):
                    window._approve_calibration_candidate()
                saved = json.loads(motion_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["reference_anchors"]["blocks"], block_pose)
                self.assertEqual(saved["reference_anchors"]["trays"], tray_pose)
                self.assertIn("已保留", window.calibration_status.text())
        finally:
            window.close()

    def test_text_submission_is_mirrored_to_voice_page_with_source_label(self) -> None:
        class FakeCompetitionWorker:
            @staticmethod
            def submit_text(text: str) -> str:
                return text.strip()

        window = CompetitionWindow()
        try:
            with patch("app.ui.CompetitionWorker", FakeCompetitionWorker):
                window.competition_worker = FakeCompetitionWorker()
                window.text_mode_button.setChecked(True)
                window.text_input_edit.setText("  请开始识别任务卡  ")
                window._send_text_instruction()
            self.assertEqual(window.voice_recognized_text.toPlainText(), "[文字控制]\n请开始识别任务卡")
            window._on_voice_interaction_finished({"action": "listen", "recognized_text": "语音指令"})
            self.assertEqual(window.voice_recognized_text.toPlainText(), "[语音识别]\n语音指令")
        finally:
            window.competition_worker = None
            window.close()


if __name__ == "__main__":
    unittest.main()
