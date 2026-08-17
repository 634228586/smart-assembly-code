from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.paths import PACKAGE_ROOT, REAL_CONFIG_DIR
from vision.offline_analyzer import (
    IMAGE_EXTENSIONS,
    OfflineAnalysisError,
    OfflineImageResult,
    analyze_image,
    default_candidate_directory,
    discover_images,
    load_scene_detector,
    normalize_detector_config,
    write_candidate,
)
from vision.workspace_localizer import COLORS


SCENE_LABELS = {"blocks": "方块 blocks", "trays": "托盘 trays"}
VIEW_LABELS = {
    "original": "原始图片",
    "mask": "六色HSV掩膜",
    "annotated": "检测标注结果",
    "returned": "最终返回点",
}
REASON_LABELS = {
    "area_below_min": "面积低于最小值",
    "area_above_max": "面积高于最大值",
    "too_small": "轮廓尺寸过小",
    "not_square": "长宽比不足（形状异常）",
    "confidence_below_min": "形状置信度低于阈值",
    "TARGET_NOT_FOUND": "未找到符合条件的该颜色目标",
    "AMBIGUOUS_TARGET": "存在多个相近候选目标",
    "multiple_candidates_best_selected": "存在多个HSV轮廓，已按面积最大优先、置信度次优返回",
    "tiny_fragments_ignored": "已忽略低于碎片噪声下限的HSV小轮廓",
    "DETECTOR_INVALID": "颜色检测参数无效",
}


class HsvRangeDialog(QDialog):
    def __init__(self, hsv_ranges: dict[str, list[dict[str, list[int]]]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑六色HSV范围")
        self.resize(850, 520)
        layout = QVBoxLayout(self)
        note = QLabel("每行是一个HSV区间；同一种颜色可添加多行。H范围0～179，S/V范围0～255。")
        layout.addWidget(note)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["颜色", "H下限", "H上限", "S下限", "S上限", "V下限", "V上限"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        row_buttons = QHBoxLayout()
        add_button = QPushButton("添加区间"); add_button.clicked.connect(lambda: self._append_row("红", [0, 0, 0], [179, 255, 255]))
        delete_button = QPushButton("删除选中区间"); delete_button.clicked.connect(self._delete_selected)
        row_buttons.addWidget(add_button); row_buttons.addWidget(delete_button); row_buttons.addStretch(1)
        layout.addLayout(row_buttons)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_and_accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for color in COLORS:
            for band in hsv_ranges.get(color, []):
                self._append_row(color, band["lower"], band["upper"])

    @staticmethod
    def _spin(maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox(); spin.setRange(0, maximum); spin.setValue(int(value)); return spin

    def _append_row(self, color: str, lower: list[int], upper: list[int]) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        combo = QComboBox(); combo.addItems(COLORS); combo.setCurrentText(color); self.table.setCellWidget(row, 0, combo)
        values = (lower[0], upper[0], lower[1], upper[1], lower[2], upper[2])
        for column, value in enumerate(values, 1):
            self.table.setCellWidget(row, column, self._spin(179 if column <= 2 else 255, int(value)))

    def _delete_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows: self.table.removeRow(row)

    def hsv_ranges(self) -> dict[str, list[dict[str, list[int]]]]:
        result: dict[str, list[dict[str, list[int]]]] = {color: [] for color in COLORS}
        for row in range(self.table.rowCount()):
            color_widget = self.table.cellWidget(row, 0)
            spins = [self.table.cellWidget(row, column) for column in range(1, 7)]
            assert isinstance(color_widget, QComboBox) and all(isinstance(widget, QSpinBox) for widget in spins)
            h1, h2, s1, s2, v1, v2 = [widget.value() for widget in spins]
            result[color_widget.currentText()].append({"lower": [h1, s1, v1], "upper": [h2, s2, v2]})
        return result

    def _validate_and_accept(self) -> None:
        ranges = self.hsv_ranges()
        missing = [color for color in COLORS if not ranges[color]]
        if missing:
            QMessageBox.warning(self, "HSV范围不完整", f"以下颜色没有任何区间：{'、'.join(missing)}"); return
        for color, bands in ranges.items():
            for index, band in enumerate(bands, 1):
                if any(lower > upper for lower, upper in zip(band["lower"], band["upper"])):
                    QMessageBox.warning(self, "HSV上下限错误", f"{color}色第{index}个区间的下限不能高于上限。"); return
        self.accept()


class OfflineVisionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("离线视觉调参与批量验证")
        self.resize(1560, 930)
        self.image_paths: list[Path] = []
        self.results: dict[Path, OfflineImageResult] = {}
        self.detector: dict[str, Any] = {}
        self.parameters_dirty = False
        self._build_ui()
        self._load_formal_detector()

    def _build_ui(self) -> None:
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        notice = QLabel(
            "离线工具：按比赛宽松规则预览——过滤极小碎片后返回面积最大的HSV轮廓；面积相同时比较置信度，"
            "面积、形状、置信度和多候选仅显示警告。"
            "只读取本地图片，不连接MVS、不连接机械臂、不写IO；九点标定严格规则不受影响。"
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("font-weight: 600; color: #7a3e00; padding: 6px; background: #fff3d6;")
        layout.addWidget(notice)

        controls = QHBoxLayout()
        self.scene_combo = QComboBox()
        for key in ("blocks", "trays"):
            self.scene_combo.addItem(SCENE_LABELS[key], key)
        self.scene_combo.currentIndexChanged.connect(self._load_formal_detector)
        controls.addWidget(QLabel("场景")); controls.addWidget(self.scene_combo)
        add_files = QPushButton("添加图片"); add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("添加文件夹（递归）"); add_folder.clicked.connect(self._add_folder)
        clear = QPushButton("清空图片"); clear.clicked.connect(self._clear_images)
        analyze = QPushButton("使用当前参数批量分析"); analyze.clicked.connect(self._analyze_all)
        analyze.setStyleSheet("font-weight: 600; background: #176f3a; color: white; padding: 6px 12px;")
        for button in (add_files, add_folder, clear, analyze): controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 5)

        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("已导入图片"))
        self.image_list = QListWidget(); self.image_list.currentItemChanged.connect(self._image_selected)
        left_layout.addWidget(self.image_list)
        self.image_count_label = QLabel("0 张")
        left_layout.addWidget(self.image_count_label)
        splitter.addWidget(left)

        center = QWidget(); center_layout = QVBoxLayout(center)
        view_controls = QHBoxLayout()
        self.view_combo = QComboBox()
        for key, label in VIEW_LABELS.items(): self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(self._refresh_image_view)
        view_controls.addWidget(QLabel("显示")); view_controls.addWidget(self.view_combo); view_controls.addStretch(1)
        center_layout.addLayout(view_controls)
        self.image_label = QLabel("请添加图片并开始分析")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(0, 0)
        self.image_label.setStyleSheet("background: #222; color: #ddd;")
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_scroll.setWidget(self.image_label)
        center_layout.addWidget(self.image_scroll)
        self.current_image_label = QLabel("尚未选择图片")
        self.current_image_label.setWordWrap(True)
        center_layout.addWidget(self.current_image_label)
        splitter.addWidget(center)

        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("当前图片六色检测详情"))
        self.detail_table = QTableWidget(0, 10)
        self.detail_table.setHorizontalHeaderLabels(
            ["颜色", "返回状态", "中心U", "中心V", "角度°", "面积", "置信度", "长宽比", "填充率", "警告"]
        )
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        self.detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.detail_table)
        splitter.addWidget(right)
        splitter.setSizes([210, 650, 700])

        parameter_group = QGroupBox("当前预览参数（HSV从正式配置或候选JSON读取；不会写正式配置）")
        parameter_group.setMaximumHeight(225)
        parameter_layout = QHBoxLayout(parameter_group)
        form = QFormLayout()
        self.roi_edit = QLineEdit()
        self.min_area_edit = QLineEdit(); self.max_area_edit = QLineEdit(); self.confidence_edit = QLineEdit()
        for editor in (self.roi_edit, self.min_area_edit, self.max_area_edit, self.confidence_edit):
            editor.textEdited.connect(self._mark_parameters_dirty)
            editor.editingFinished.connect(self._auto_apply_parameters)
        form.addRow("ROI x1,y1,x2,y2", self.roi_edit)
        form.addRow("最小面积 px²", self.min_area_edit)
        form.addRow("最大面积 px²", self.max_area_edit)
        form.addRow("最低形状置信度", self.confidence_edit)
        parameter_layout.addLayout(form, 2)
        hsv_box = QVBoxLayout()
        hsv_box.addWidget(QLabel("当前六色HSV范围（只读显示）"))
        self.hsv_ranges_display = QPlainTextEdit()
        self.hsv_ranges_display.setReadOnly(True)
        self.hsv_ranges_display.setMaximumHeight(165)
        self.hsv_ranges_display.setStyleSheet("font-family: Consolas, 'Microsoft YaHei'; background: #f7f7f7;")
        hsv_box.addWidget(self.hsv_ranges_display)
        parameter_layout.addLayout(hsv_box, 3)
        buttons = QVBoxLayout()
        reload_formal = QPushButton("重新载入正式 camera.json"); reload_formal.clicked.connect(self._load_formal_detector)
        load_candidate = QPushButton("加载候选JSON"); load_candidate.clicked.connect(self._load_candidate)
        self.edit_hsv_button = QPushButton("编辑六色HSV范围"); self.edit_hsv_button.clicked.connect(self._edit_hsv_ranges)
        apply_preview = QPushButton("应用参数到预览"); apply_preview.clicked.connect(self._apply_preview_parameters)
        save_candidate_button = QPushButton("保存候选JSON"); save_candidate_button.clicked.connect(self._save_candidate)
        for button in (reload_formal, load_candidate, self.edit_hsv_button, apply_preview, save_candidate_button): buttons.addWidget(button)
        buttons.addStretch(1); parameter_layout.addLayout(buttons)
        layout.addWidget(parameter_group)

        layout.addWidget(QLabel("整批图片验证结果"))
        self.batch_table = QTableWidget(0, 8)
        self.batch_table.setHorizontalHeaderLabels(["图片", *COLORS, "总体"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.batch_table.cellClicked.connect(self._batch_cell_clicked)
        layout.addWidget(self.batch_table, 2)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    @property
    def scene(self) -> str:
        return str(self.scene_combo.currentData())

    def _set_detector_fields(self, detector: dict[str, Any]) -> None:
        self.detector = normalize_detector_config(detector)
        self.roi_edit.setText(",".join(str(value) for value in self.detector["roi"]))
        self.min_area_edit.setText(str(self.detector["min_area_px"]))
        self.max_area_edit.setText(str(self.detector["max_area_px"]))
        self.confidence_edit.setText(str(self.detector["confidence_min"]))
        self._refresh_hsv_ranges_display()
        self.parameters_dirty = False
        self.results.clear(); self._clear_result_views()

    def _refresh_hsv_ranges_display(self) -> None:
        lines: list[str] = []
        for color in COLORS:
            bands = self.detector.get("hsv_ranges", {}).get(color, [])
            values = [
                f"H {band['lower'][0]}–{band['upper'][0]}  S {band['lower'][1]}–{band['upper'][1]}  V {band['lower'][2]}–{band['upper'][2]}"
                for band in bands
            ]
            lines.append(f"{color}：" + "  |  ".join(values))
        self.hsv_ranges_display.setPlainText("\n".join(lines))

    def _load_formal_detector(self) -> None:
        try:
            self._set_detector_fields(load_scene_detector(REAL_CONFIG_DIR / "camera.json", self.scene))
            self.status_label.setText(f"已只读载入正式 camera.json 中的 {self.scene} 参数。")
        except OfflineAnalysisError as exc:
            QMessageBox.critical(self, "载入失败", str(exc))

    def _detector_from_fields(self) -> dict[str, Any]:
        try:
            roi = [int(part.strip()) for part in self.roi_edit.text().split(",")]
        except ValueError as exc:
            raise OfflineAnalysisError(f"ROI格式错误：{exc}") from exc
        hsv = self.detector.get("hsv_ranges")
        return normalize_detector_config({
            "roi": roi,
            "confidence_min": self.confidence_edit.text(),
            "min_area_px": self.min_area_edit.text(),
            "max_area_px": self.max_area_edit.text(),
            "hsv_ranges": hsv,
        })

    def _apply_preview_parameters(self) -> None:
        try:
            self.detector = self._detector_from_fields()
        except OfflineAnalysisError as exc:
            QMessageBox.warning(self, "参数无效", str(exc)); return
        self.parameters_dirty = False
        self.results.clear(); self._clear_result_views()
        self.status_label.setText("参数已应用到内存预览；尚未分析，且未写入任何文件。")
        if self.image_paths:
            self._analyze_all()

    def _mark_parameters_dirty(self, _text: str = "") -> None:
        if self.parameters_dirty:
            return
        self.parameters_dirty = True
        self.results.clear(); self._clear_result_views()
        self.status_label.setText("预览参数已修改，旧分析结果已作废；离开输入框或按回车后自动重新分析。")

    def _auto_apply_parameters(self) -> None:
        if not self.parameters_dirty:
            return
        try:
            self.detector = self._detector_from_fields()
        except OfflineAnalysisError as exc:
            self.status_label.setText(f"预览参数尚未生效：{exc}")
            return
        self.parameters_dirty = False
        self.status_label.setText("新范围已应用到内存预览；正式 camera.json 未修改。")
        if self.image_paths:
            self._analyze_all()

    def _edit_hsv_ranges(self) -> None:
        try:
            detector = self._detector_from_fields()
        except OfflineAnalysisError as exc:
            QMessageBox.warning(self, "参数无效", str(exc)); return
        dialog = HsvRangeDialog(detector["hsv_ranges"], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        detector["hsv_ranges"] = dialog.hsv_ranges()
        try:
            self._set_detector_fields(detector)
        except OfflineAnalysisError as exc:
            QMessageBox.warning(self, "HSV范围无效", str(exc)); return
        self.status_label.setText("新的六色HSV范围已应用到内存预览；正式 camera.json 未修改。")
        if self.image_paths:
            self._analyze_all()

    def _add_paths(self, paths: list[str | Path]) -> None:
        additions = discover_images(paths)
        existing = {str(path).casefold() for path in self.image_paths}
        for path in additions:
            if str(path).casefold() in existing: continue
            self.image_paths.append(path); existing.add(str(path).casefold())
            item = QListWidgetItem(path.name); item.setData(Qt.ItemDataRole.UserRole, str(path)); item.setToolTip(str(path))
            self.image_list.addItem(item)
        self.image_count_label.setText(f"{len(self.image_paths)} 张")
        if self.image_list.currentRow() < 0 and self.image_list.count(): self.image_list.setCurrentRow(0)
        self.status_label.setText(f"已导入 {len(self.image_paths)} 张图片；等待批量分析。")

    def _add_files(self) -> None:
        filters = "图片 (" + " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS)) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "选择采集图片", str(PACKAGE_ROOT / "data"), filters)
        if paths: self._add_paths(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择包含采集图片的文件夹", str(PACKAGE_ROOT / "data"))
        if path: self._add_paths([path])

    def _clear_images(self) -> None:
        self.image_paths.clear(); self.results.clear(); self.image_list.clear(); self.batch_table.setRowCount(0)
        self.image_count_label.setText("0 张"); self._clear_result_views(); self.status_label.setText("图片已清空。")

    def _clear_result_views(self) -> None:
        self.detail_table.setRowCount(0); self.batch_table.setRowCount(0)
        self.image_label.clear(); self.image_label.setText("请选择图片并开始分析")

    def _analyze_all(self) -> None:
        if not self.image_paths:
            QMessageBox.information(self, "没有图片", "请先添加采集图片或文件夹。"); return
        try:
            self.detector = self._detector_from_fields()
        except OfflineAnalysisError as exc:
            QMessageBox.warning(self, "参数无效", str(exc)); return
        self.parameters_dirty = False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        failures: list[str] = []
        try:
            self.results.clear()
            for index, path in enumerate(self.image_paths, 1):
                self.status_label.setText(f"正在分析 {index}/{len(self.image_paths)}：{path.name}")
                QApplication.processEvents()
                try:
                    self.results[path] = analyze_image(path, self.detector)
                except OfflineAnalysisError as exc:
                    failures.append(f"{path.name}: {exc}")
            self._populate_batch_table()
            self._image_selected(self.image_list.currentItem(), None)
        finally:
            QApplication.restoreOverrideCursor()
        passed = sum(1 for result in self.results.values() if result.summary.get("success") is True)
        self.status_label.setText(
            f"分析完成：{len(self.results)}张成功读取，{passed}张六色均有返回，{len(failures)}张分析异常。"
        )
        if failures:
            QMessageBox.warning(self, "部分图片分析失败", "\n".join(failures[:20]))

    def _populate_batch_table(self) -> None:
        self.batch_table.setRowCount(len(self.image_paths))
        for row, path in enumerate(self.image_paths):
            first = QTableWidgetItem(path.name); first.setToolTip(str(path)); self.batch_table.setItem(row, 0, first)
            result = self.results.get(path)
            reports = {str(report["color"]): report for report in result.summary.get("colors", [])} if result else {}
            for column, color in enumerate(COLORS, 1):
                report = reports.get(color)
                status = str(report.get("status")) if report else "未分析"
                item = QTableWidgetItem({"success": "已返回", "not_found": "无HSV轮廓", "ambiguous": "多候选"}.get(status, status))
                item.setBackground(QColor("#d8f2df") if status == "success" else QColor("#ffd9d9"))
                self.batch_table.setItem(row, column, item)
            overall_ok = bool(result and result.summary.get("success") is True)
            overall = QTableWidgetItem("六色均返回" if overall_ok else "存在无轮廓颜色")
            overall.setBackground(QColor("#bde8c7") if overall_ok else QColor("#ffbcbc"))
            self.batch_table.setItem(row, 7, overall)
        self.batch_table.resizeColumnsToContents()

    def _batch_cell_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < self.image_list.count(): self.image_list.setCurrentRow(row)

    def _selected_path(self) -> Path | None:
        item = self.image_list.currentItem()
        return Path(str(item.data(Qt.ItemDataRole.UserRole))) if item else None

    def _image_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None: return
        path = Path(str(current.data(Qt.ItemDataRole.UserRole)))
        self.current_image_label.setText(str(path))
        result = self.results.get(path)
        if result is None:
            self.detail_table.setRowCount(0); self.image_label.setText("该图片尚未分析")
            return
        self._populate_detail(result); self._refresh_image_view()

    def _populate_detail(self, result: OfflineImageResult) -> None:
        reports = list(result.summary.get("colors", [])); self.detail_table.setRowCount(len(reports))
        for row, report in enumerate(reports):
            selected = report.get("selected")
            candidates = report.get("candidates") or []
            best = selected if isinstance(selected, dict) else (candidates[0] if candidates else {})
            reason_codes = list(report.get("warnings") or [])
            reason = "；".join(REASON_LABELS.get(str(code), str(code)) for code in reason_codes)
            if not reason:
                error_code = report.get("error_code")
                reason = REASON_LABELS.get(str(error_code), str(error_code)) if error_code else ""
            center = best.get("center", []) if isinstance(best, dict) else []
            values = [
                str(report.get("color", "")), "已返回" if report.get("status") == "success" else "无HSV轮廓",
                f"{float(center[0]):.1f}" if len(center) == 2 else "-",
                f"{float(center[1]):.1f}" if len(center) == 2 else "-",
                f"{float(best.get('angle_deg', 0)):.2f}" if best else "-",
                f"{float(best.get('area_px', 0)):.0f}" if best else "-",
                f"{float(best.get('confidence', 0)):.3f}" if best else "-",
                f"{float(best.get('aspect', 0)):.3f}" if best else "-",
                f"{float(best.get('rectangularity', 0)):.3f}" if best else "-",
                reason or "无",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(QColor("#d8f2df") if report.get("status") == "success" else QColor("#ffd9d9"))
                self.detail_table.setItem(row, column, item)
        self.detail_table.resizeColumnsToContents()

    @staticmethod
    def _pixmap(image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _refresh_image_view(self) -> None:
        path = self._selected_path(); result = self.results.get(path) if path else None
        if result is None: return
        mode = str(self.view_combo.currentData())
        image = {
            "original": result.original_bgr,
            "mask": result.mask_bgr,
            "annotated": result.annotated_bgr,
            "returned": result.returned_bgr,
        }[mode]
        pixmap = self._pixmap(image)
        target = self.image_scroll.viewport().size()
        self.image_label.setPixmap(pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_image_view()

    def _load_candidate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载离线候选参数", str(default_candidate_directory()), "JSON (*.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("kind") != "offline_detector_candidate" or payload.get("scene") not in {"blocks", "trays"}:
                raise OfflineAnalysisError("不是有效的离线检测候选文件。")
            index = self.scene_combo.findData(payload["scene"])
            self.scene_combo.blockSignals(True); self.scene_combo.setCurrentIndex(index); self.scene_combo.blockSignals(False)
            self._set_detector_fields(payload["detector"])
            self.status_label.setText(f"已载入候选参数：{path}；尚未写入正式配置。")
        except (OSError, json.JSONDecodeError, KeyError, OfflineAnalysisError) as exc:
            QMessageBox.warning(self, "候选载入失败", str(exc))

    def _save_candidate(self) -> None:
        try:
            detector = self._detector_from_fields()
        except OfflineAnalysisError as exc:
            QMessageBox.warning(self, "参数无效", str(exc)); return
        default_candidate_directory().mkdir(parents=True, exist_ok=True)
        default_name = f"{self.scene}_detector_candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "保存候选参数", str(default_candidate_directory() / default_name), "JSON (*.json)")
        if not path: return
        try:
            saved = write_candidate(path, scene=self.scene, detector=detector, source_images=self.image_paths)
        except OfflineAnalysisError as exc:
            QMessageBox.critical(self, "保存失败", str(exc)); return
        self.status_label.setText(f"候选参数已保存：{saved}；正式 camera.json 未修改。")


def run_offline_vision_tool() -> int:
    app = QApplication.instance() or QApplication([])
    window = OfflineVisionWindow(); window.show()
    return app.exec()
