from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.offline_vision_tool import HsvRangeDialog, OfflineVisionWindow, REASON_LABELS, TEMP_TEST_HSV_RANGES
from tests.test_offline_analyzer import detector, write_synthetic


class OfflineVisionUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_import_analyze_and_switch_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"; write_synthetic(path)
            window = OfflineVisionWindow()
            try:
                window._set_detector_fields(detector())
                window._add_paths([path])
                window._analyze_all()
                self.app.processEvents()
                self.assertEqual(window.image_list.count(), 1)
                self.assertEqual(window.batch_table.rowCount(), 1)
                self.assertEqual(window.detail_table.rowCount(), 6)
                self.assertEqual(window.batch_table.item(0, 7).text(), "六色均返回")
                self.assertIn(path.resolve(), window.results)
                self.assertFalse(hasattr(window, "hsv_edit"))
                self.assertIn("编辑六色HSV范围", window.edit_hsv_button.text())
                self.assertIn("红：H", window.hsv_ranges_display.toPlainText())
                self.assertEqual(window.detail_table.item(0, 1).text(), "已返回")
                self.assertNotEqual(window.detail_table.item(0, 2).text(), "-")
                self.assertNotEqual(window.detail_table.item(0, 3).text(), "-")
                for mode in ("original", "mask", "annotated", "returned"):
                    window.view_combo.setCurrentIndex(window.view_combo.findData(mode))
                    self.app.processEvents()
                    pixmap = window.image_label.pixmap()
                    self.assertIsNotNone(pixmap)
                    self.assertLessEqual(pixmap.width(), window.image_scroll.viewport().width())
                    self.assertLessEqual(pixmap.height(), window.image_scroll.viewport().height())
            finally:
                window.close()

    def test_internal_rejection_reasons_have_chinese_labels(self) -> None:
        self.assertEqual(REASON_LABELS["area_above_max"], "面积高于最大值")
        self.assertEqual(REASON_LABELS["TARGET_NOT_FOUND"], "未找到符合条件的该颜色目标")

    def test_startup_uses_temporary_test_hsv_ranges(self) -> None:
        window = OfflineVisionWindow()
        try:
            self.assertEqual(window.detector["hsv_ranges"], TEMP_TEST_HSV_RANGES)
        finally:
            window.close()

    def test_hsv_table_round_trips_multiple_ranges(self) -> None:
        ranges = detector()["hsv_ranges"]
        ranges["红"].append({"lower": [170, 150, 100], "upper": [179, 255, 255]})
        dialog = HsvRangeDialog(ranges)
        try:
            self.assertEqual(dialog.table.rowCount(), 7)
            self.assertEqual(dialog.hsv_ranges(), ranges)
            h_lower = dialog.table.cellWidget(0, 1)
            h_lower.setValue(1)
            self.assertEqual(dialog.hsv_ranges()["红"][0]["lower"][0], 1)
        finally:
            dialog.close()

    def test_editing_area_range_invalidates_cache_and_reanalyzes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"; write_synthetic(path)
            window = OfflineVisionWindow()
            try:
                window._set_detector_fields(detector())
                window._add_paths([path]); window._analyze_all()
                self.assertTrue(window.results[path.resolve()].summary["success"])
                window.min_area_edit.setText("700")
                window.min_area_edit.textEdited.emit("700")
                self.assertTrue(window.parameters_dirty)
                self.assertEqual(window.results, {})
                window.min_area_edit.editingFinished.emit()
                self.assertFalse(window.parameters_dirty)
                self.assertEqual(window.detector["min_area_px"], 700.0)
                result = window.results[path.resolve()]
                self.assertTrue(result.summary["success"])
                self.assertTrue(all("area_below_min" in report["warnings"] for report in result.summary["colors"]))
                self.assertIn("面积低于最小值", window.detail_table.item(0, 9).text())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
