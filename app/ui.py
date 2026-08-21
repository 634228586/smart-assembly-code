from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import load_all
from .camera_config_editor import (
    PROFILE_KEYS,
    PROFILE_LABELS,
    CameraConfigInputError,
    approve_detector_values,
    approve_profile_batch,
    approve_profile_batch_by_operator,
    load_detector_editor_values,
    load_camera_editor_values,
    save_detector_editor_values,
    save_camera_editor_values,
)
from .competition_worker import CompetitionWorker
from .direct_assembly_worker import DirectAssemblyWorker
from .io_worker import ManualIoWorker
from .point_capture_worker import RobotPointCaptureWorker, RobotPointMoveWorker, RobotReferenceAnchorCaptureWorker, maintenance_point_limits
from .calibration_config_editor import CalibrationConfigError, COLORS, load_calibration_settings, mark_automatic_verified, save_calibration_settings
from .calibration_worker import CalibrationValidationWorker, CalibrationWorker, CurrentMvsParametersWorker, DetectorAreaEstimateWorker, DetectorValidationWorker, ManualSceneCaptureWorker, ManualSceneRecognitionWorker, ProfileValidationWorker, RobotReadinessWorker
from .nine_point import SCENES, approve_candidate_without_direction_validation, build_grid
from .paths import EVIDENCE_DIR, REAL_CALIBRATION_DIR, REAL_CONFIG_DIR, resolve_project_path
from .preflight import Check, competition_ready, run_static_preflight
from .robot_config_editor import (
    POINT_KEYS,
    POINT_LABELS,
    RobotConfigInputError,
    convert_joint_display,
    load_reference_anchors,
    load_robot_editor_values,
    save_robot_editor_values,
)
from .session import CompetitionSession, SessionError
from .task_card_model_test_worker import TaskCardModelTestWorker
from .voice_interaction_worker import VoiceInteractionWorker


PREFLIGHT_FRIENDLY: dict[str, tuple[str, str, str]] = {
    "完整配置集": ("正式配置文件能否正常读取", "确认比赛所需的全部配置文件存在、格式正确且没有损坏。", "正式目录的 config/real 文件夹"),
    "endpoints": ("机器人、视觉和语音连接地址是否填写完整", "检查各设备的 IP、端口和服务身份；缺失时程序不知道应该连接谁。", "config/real/endpoints.json；地址必须从真实设备现场确认"),
    "robot": ("真实机器人身份、活动TCP和负载是否填写完整", "检查唯一机器人名称/序列号、登录信息、活动TCP、负载及末端安装确认。", "“机器人现场配置”页和真实 ARCS"),
    "camera": ("MVS相机和三套拍照参数是否填写完整", "检查任务卡/方块/托盘三套参数及两套颜色检测参数。", "“相机与视觉”页"),
    "motion": ("拍照点、六色抓放基准、速度和九点参数是否填写完整", "检查四个固定关节点、Block/Tray各六个完整TCP、比赛速度以及九点速度/步长。", "“机器人现场配置”和“维护与真实九点标定”页"),
    "suction_io": ("真实吸盘IO端口和有效电平是否确认", "检查吸盘使用哪个真实IO、吸取/释放电平及反馈输入，禁止猜测端口。", "真实控制柜接线与 config/real/suction_io.json"),
    "competition": ("比赛流程规则配置是否完整", "检查双任务卡、识别失败和停止撤权等比赛规则是否齐全。", "正式比赛配置；当前通常无需现场修改"),
    "角色与监听冲突": ("各服务端口是否分工正确且没有冲突", "确认机器人、MVS、语音和Qt监听端口没有被调换或重复占用。", "config/real/endpoints.json"),
    "MVS Runtime": ("海康MVS运行库是否已安装", "真实工业相机需要厂商运行库才能被程序打开。", "Windows已安装的海康MVS软件/Runtime"),
    "包内 MVS Python wrapper": ("比赛包内是否带有MVS接口文件", "确认复制整个比赛文件夹后，程序仍有调用相机SDK所需的Python接口。", "正式目录 vision/vendor/mvs"),
    "包内真实 MVS 服务": ("真实MVS视觉服务程序是否存在", "确认比赛使用真实相机服务，而不是固定图片或模拟识别。", "正式目录的“启动MVS视觉服务.cmd”"),
    "工具安装核验": ("末端相机和吸盘安装是否已现场确认", "确认眼在手相机、圆形吸盘和工具方向与配置一致。", "真实机械臂末端检查；确认后保存现场证据"),
    "负载核验": ("机械臂负载和重心是否已在ARCS确认", "负载或重心错误会影响运动控制和安全。", "真实 ARCS 的负载/重心设置"),
    "DO0光圈配置": ("光圈是否固定配置为DO0", "检查光圈使用DO0，且False为关闭、True为打开。", "已确认接线映射与 config/real/suction_io.json"),
    "三套采集参数批准": ("任务卡、方块、托盘三套相机参数是否已由操作者确认", "现场操作者确认保存值后即可人工批准；程序不要求连接相机回读。运行时只按保存值写入。", "“相机与视觉”页的人工批准按钮"),
    "两套颜色参数完整": ("方块和托盘的颜色识别参数是否填写完整", "检查两套ROI、面积、置信度和六色HSV是否齐全；不再要求六色同帧批准。", "使用 Block/Tray 拍照取图，离线分析后直接更新 config/real/camera.json，并重启MVS服务"),
    "直接 MoveJoint策略": ("是否采用已确认的直接MoveJoint比赛策略", "程序不做路线验收和碰撞规划；每场必须由操作者确认路径无人且无障碍物。", "“比赛任务”页的全局直达路径安全确认"),
    "真实机器人总放行": ("真实机器人是否完成最终分级验收", "代表点位、速度、TCP、下降、IO和装夹流程已经按阶段实际验证。", "完成真机逐级验收后更新放行记录"),
    "blocks": ("方块工作区九点标定是否有效", "必须存在一套最新批准的方块标定，并与当前相机、机器人、拍照点、TCP和分辨率一致。", "“维护与真实九点标定”页选择 blocks"),
    "trays": ("托盘工作区九点标定是否有效", "必须存在一套最新批准的托盘标定，并与当前相机、机器人、拍照点、TCP和分辨率一致。", "“维护与真实九点标定”页选择 trays"),
    "包路径独立": ("比赛文件夹是否可以整体复制运行", "确认程序从内层“装配赛正式代码”目录运行，没有依赖旧开发工程。", "从正式目录启动程序"),
}


def _friendly_preflight(check: Check) -> tuple[str, str, str]:
    if check.item in PREFLIGHT_FRIENDLY:
        return PREFLIGHT_FRIENDLY[check.item]
    if check.item in {"PySide6", "pyaubo_sdk", "cv2", "yaml", "openai"}:
        return (f"程序组件 {check.item} 是否可用", f"正式程序运行所需的 {check.item} 组件必须能正常加载。", "包内 .runtime；缺失时重建包内运行环境")
    if check.item.startswith("DASHSCOPE_"):
        label = {"DASHSCOPE_API_KEY": "大模型密钥", "DASHSCOPE_BASE_URL": "大模型服务地址", "DASHSCOPE_MODEL": "大模型名称"}.get(check.item, check.item)
        return (f"{label}是否已设置", "任务卡需要上传给大模型识别；该环境变量缺失时不能完成真实识别。", "当前Windows用户环境变量（不要写进比赛文件夹）")
    return (check.item, "检查这一项是否达到正式比赛的安全和运行要求。", check.action or "按现场检查要求处理")


def _friendly_action(check: Check) -> str:
    specific = {
        "endpoints": "从真实 ARCS、MVS服务和AI盒子确认各自IP/端口后写入配置；任何地址都不能猜测或互换。",
        "robot": "先在真实 ARCS确认机器人身份、活动TCP和负载，再把对应值填入“机器人现场配置”。",
        "camera": "在“相机与视觉”填写三套采集参数；使用 Block/Tray 拍照取图，离线分析后直接更新两套颜色参数。",
        "motion": "在“机器人现场配置”填写固定关节点并保存两个完整红色抓放TCP，在九点页填写实际速度、加速度和步长。",
        "suction_io": "查看真实控制柜接线或设备电气资料，确认输出编号和有效电平后再写入；没有依据时保持未配置。",
    }
    return specific.get(check.item, check.action or "当前已经满足，无需处理。")


class CompetitionWindow(QMainWindow):
    """正式比赛壳层；当前里程碑只提供失效安全检查，不连接硬件。"""

    def __init__(self) -> None:
        super().__init__()
        self.session = CompetitionSession()
        self.checks: list[Check] = []
        self.competition_thread: QThread | None = None
        self.competition_worker: CompetitionWorker | DirectAssemblyWorker | None = None
        self.calibration_thread: QThread | None = None
        self.calibration_worker: CalibrationWorker | None = None
        self.validation_thread: QThread | None = None
        self.validation_worker: CalibrationValidationWorker | None = None
        self.detector_thread: QThread | None = None
        self.detector_worker: DetectorValidationWorker | DetectorAreaEstimateWorker | ManualSceneCaptureWorker | ManualSceneRecognitionWorker | None = None
        self.profile_thread: QThread | None = None
        self.profile_worker: ProfileValidationWorker | TaskCardModelTestWorker | None = None
        self.mvs_read_thread: QThread | None = None
        self.mvs_read_worker: CurrentMvsParametersWorker | None = None
        self.readiness_thread: QThread | None = None
        self.readiness_worker: RobotReadinessWorker | None = None
        self.io_thread: QThread | None = None
        self.io_worker: ManualIoWorker | None = None
        self.point_capture_thread: QThread | None = None
        self.point_capture_worker: RobotPointCaptureWorker | None = None
        self.point_move_thread: QThread | None = None
        self.point_move_worker: RobotPointMoveWorker | None = None
        self.contact_capture_thread: QThread | None = None
        self.contact_capture_worker: RobotReferenceAnchorCaptureWorker | None = None
        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceInteractionWorker | None = None
        self.calibration_candidate: dict[str, object] | None = None
        self.calibration_validation: dict[str, object] = {}
        self.setWindowTitle("装配赛正式总控｜真实硬件门控")
        self.resize(1450, 900)
        self.setMinimumSize(1100, 720)
        self._build_ui()
        self.run_preflight()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        banner = QLabel("正式真实机械臂版本｜缺项即禁用｜不会自动连接、上电、startup、运动或写 IO")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet("background:#6b1010;color:white;font-weight:700;padding:10px;")
        layout.addWidget(banner)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._preflight_page(), "赛前检查")
        self.tabs.addTab(self._task_page(), "比赛任务")
        self.tabs.addTab(self._monitor_page(), "自动流程监控")
        self.tabs.addTab(self._direct_assembly_page(), "单组抓放调试")
        self.tabs.addTab(self._camera_config_page(), "相机与视觉")
        self.tabs.addTab(self._robot_config_page(), "机器人现场配置")
        self.tabs.addTab(self._manual_io_page(), "IO单步调试")
        self.tabs.addTab(self._calibration_page(), "维护与真实九点标定")
        self.tabs.addTab(self._voice_interaction_page(), "语音盒子测试")
        self.tabs.addTab(self._logs_page(), "日志与证据")
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _preflight_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.preflight_status = QLabel("尚未检查")
        self.preflight_status.setStyleSheet("font-size:16px;font-weight:700;")
        run_button = QPushButton("重新执行静态赛前检查")
        run_button.clicked.connect(self.run_preflight)
        controls.addWidget(self.preflight_status); controls.addStretch(); controls.addWidget(run_button)
        layout.addLayout(controls)
        self.preflight_table = QTableWidget(0, 7)
        self.preflight_table.setHorizontalHeaderLabels(("类别", "检查内容（通俗说明）", "当前情况", "合格标准", "结果", "去哪里处理", "是否阻止比赛"))
        self.preflight_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preflight_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.preflight_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.preflight_table.setWordWrap(False)
        self.preflight_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preflight_table.verticalHeader().setVisible(False)
        header = self.preflight_table.horizontalHeader()
        for column in (0, 1, 4, 6): header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3, 5): header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        self.preflight_table.setColumnWidth(2, 360); self.preflight_table.setColumnWidth(3, 300); self.preflight_table.setColumnWidth(5, 360)
        self.preflight_table.itemSelectionChanged.connect(self._update_preflight_detail)
        layout.addWidget(self.preflight_table)
        detail_group = QGroupBox("选中检查项的完整内容（表格也可以横向滚动，鼠标悬停可看全文）")
        detail_layout = QVBoxLayout(detail_group)
        self.preflight_detail = QPlainTextEdit(); self.preflight_detail.setReadOnly(True); self.preflight_detail.setMaximumHeight(150)
        self.preflight_detail.setPlainText("请选择一行查看完整内容。")
        copy_detail = QPushButton("复制当前选中项完整内容"); copy_detail.clicked.connect(self._copy_preflight_detail)
        detail_layout.addWidget(self.preflight_detail); detail_layout.addWidget(copy_detail)
        layout.addWidget(detail_group)
        return page

    def _update_preflight_detail(self) -> None:
        row = self.preflight_table.currentRow()
        if not 0 <= row < len(self.checks):
            self.preflight_detail.setPlainText("请选择一行查看完整内容。"); return
        check = self.checks[row]
        title, explanation, location = _friendly_preflight(check)
        status = {"PASS": "【通过】", "FAIL": "【未通过】", "WARN": "【提醒】"}.get(check.status, check.status)
        self.preflight_detail.setPlainText(
            f"检查内容：{title}\n检查结果：{status}\n是否阻止比赛：{'是，未通过时不能授权比赛' if check.critical else '否，仅作提醒'}\n\n"
            f"这项在检查什么：{explanation}\n\n当前发现：{check.actual}\n合格标准：{check.expected}\n"
            f"你需要怎么做：{_friendly_action(check)}\n去哪里处理：{location}\n\n技术标识：{check.category} / {check.item}"
        )

    def _copy_preflight_detail(self) -> None:
        QApplication.clipboard().setText(self.preflight_detail.toPlainText())

    def _task_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        gate = QGroupBox("本场比赛自动执行授权")
        form = QFormLayout(gate)
        self.authorization_check = QCheckBox("我已完成全部赛前检查，授权本场双任务卡自动流程")
        self.authorization_check.stateChanged.connect(self._refresh_authorization_button)
        self.direct_paths_check = QCheckBox("我已确认当前姿态及任务卡/方块/托盘全部直接 MoveJoint路径无人员和障碍物；程序不提供碰撞规划")
        self.direct_paths_check.stateChanged.connect(self._refresh_authorization_button)
        self.authorize_button = QPushButton("建立一次性比赛会话授权")
        self.authorize_button.clicked.connect(self._authorize)
        self.authorize_button.setEnabled(False)
        self.session_status = QLabel("未授权")
        form.addRow(self.authorization_check)
        form.addRow(self.direct_paths_check)
        form.addRow("会话状态", self.session_status)
        form.addRow(self.authorize_button)
        execution = QGroupBox("正式流程（仅在本场授权后可用）")
        execution_layout = QHBoxLayout(execution)
        self.start_competition_button = QPushButton("启动真实双任务卡比赛流程")
        self.start_competition_button.setEnabled(False)
        self.start_competition_button.clicked.connect(self._start_competition)
        self.stop_competition_button = QPushButton("人工停止并撤销授权")
        self.stop_competition_button.setEnabled(False)
        self.stop_competition_button.clicked.connect(self._stop_competition)
        execution_layout.addWidget(self.start_competition_button)
        execution_layout.addWidget(self.stop_competition_button)
        layout.addWidget(execution)
        layout.addWidget(gate)
        text_control = QGroupBox("语音 / 文字 / 倒计时控制切换")
        text_layout = QGridLayout(text_control)
        self.text_mode_button = QPushButton("切换到文字控制")
        self.text_mode_button.setCheckable(True)
        self.text_mode_button.toggled.connect(self._toggle_text_mode)
        self.countdown_mode_button = QPushButton("切换到5秒倒计时控制")
        self.countdown_mode_button.setCheckable(True)
        self.countdown_mode_button.toggled.connect(self._toggle_countdown_mode)
        self.text_input_edit = QLineEdit()
        self.text_input_edit.setPlaceholderText("可先输入“小具同学”；建立授权并启动流程后再发送")
        self.text_input_edit.returnPressed.connect(self._send_text_instruction)
        self.send_text_button = QPushButton("发送文字指令")
        self.send_text_button.clicked.connect(self._send_text_instruction)
        self.text_control_status = QLabel("当前：语音控制。比赛启动前可切换为文字控制。")
        self.text_control_status.setWordWrap(True)
        text_layout.addWidget(self.text_mode_button, 0, 0)
        text_layout.addWidget(self.countdown_mode_button, 0, 1, 1, 2)
        text_layout.addWidget(self.text_input_edit, 1, 0, 1, 2)
        text_layout.addWidget(self.send_text_button, 1, 2)
        text_layout.addWidget(self.text_control_status, 2, 0, 1, 3)
        layout.addWidget(text_control)
        self._refresh_text_controls()

        model_test = QGroupBox("任务卡拍照 → 大模型正式数据测试（只测试，不执行）")
        model_layout = QGridLayout(model_test)
        model_note = QLabel(
            "使用task_card当前采集参数拍摄一张全新照片，并调用正式Qwen提示词、正式封装器和正式协议校验器。"
            "图片会发送到当前配置的大模型服务，可能产生调用费用；不会连接机器人、写IO或进入比赛会话。"
        )
        model_note.setWordWrap(True); model_note.setStyleSheet("padding:8px;background:#e8f1ff;font-weight:700;")
        self.task_model_test_button = QPushButton("拍摄任务卡并发送给大模型测试")
        self.task_model_test_button.setMinimumHeight(42)
        self.task_model_test_button.clicked.connect(self._start_task_card_model_test)
        self.task_model_test_status = QLabel("尚未测试。")
        self.task_model_test_status.setWordWrap(True); self.task_model_test_status.setStyleSheet("padding:8px;background:#f1f1f1;font-weight:700;")
        self.task_model_test_image = QLabel("本次任务卡新图将在这里显示")
        self.task_model_test_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.task_model_test_image.setMinimumHeight(150)
        self.task_model_test_image.setStyleSheet("background:#222;color:#ddd;padding:8px;")
        self.task_model_raw_text = QPlainTextEdit(); self.task_model_raw_text.setReadOnly(True)
        self.task_model_raw_text.setPlaceholderText("模型原始返回将在这里显示")
        self.task_model_raw_text.setMinimumHeight(130)
        self.task_model_formal_text = QPlainTextEdit(); self.task_model_formal_text.setReadOnly(True)
        self.task_model_formal_text.setPlaceholderText("解析、正式封装和校验结果将在这里显示")
        self.task_model_formal_text.setMinimumHeight(130)
        model_layout.addWidget(model_note, 0, 0, 1, 2)
        model_layout.addWidget(self.task_model_test_button, 1, 0, 1, 2)
        model_layout.addWidget(self.task_model_test_status, 2, 0, 1, 2)
        model_layout.addWidget(self.task_model_test_image, 3, 0, 1, 2)
        model_layout.addWidget(QLabel("大模型原始返回"), 4, 0)
        model_layout.addWidget(QLabel("解析结果 / 正式协议结果"), 4, 1)
        model_layout.addWidget(self.task_model_raw_text, 5, 0)
        model_layout.addWidget(self.task_model_formal_text, 5, 1)
        layout.addWidget(model_test)

        task_info = QLabel(
            "输入规则：语音和文字模式均首次需要“小具同学”唤醒；倒计时模式在到达任务卡拍照点后自动按5秒+5秒触发，第二张任务卡等待12秒自动识别。\n"
            "任务一卡只播报和记录；任务卡二收到合法六组数据后，在全部硬件门控有效时立即装夹。\n"
            "TTS失败为黄色警告，不阻止已验证的任务二执行；识别失败必须明确告知且绝不运动。"
        )
        task_info.setWordWrap(True); task_info.setStyleSheet("padding:12px;background:#fff4cc;")
        layout.addWidget(task_info); layout.addStretch()
        return page

    def _start_task_card_model_test(self) -> None:
        if any(thread is not None for thread in (
            self.competition_thread, self.calibration_thread, self.validation_thread,
            self.detector_thread, self.profile_thread, self.mvs_read_thread,
            self.readiness_thread, self.io_thread, self.point_capture_thread,
            self.point_move_thread, self.contact_capture_thread, self.voice_thread,
        )):
            QMessageBox.critical(self, "不能测试", "比赛流程或其他硬件/服务测试正在运行。")
            return
        answer = QMessageBox.question(
            self, "确认拍照并发送给大模型",
            "将连接真实MVS视觉服务，使用task_card参数软件触发一张新图，并把该图片发送到当前配置的Qwen大模型。\n\n"
            "这可能产生模型调用费用；不会连接机器人、运动、写IO或执行模型返回的任务。请确认相机已在任务卡拍照位置，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.profile_thread = QThread(self); self.profile_worker = TaskCardModelTestWorker()
        self.profile_worker.moveToThread(self.profile_thread); self.profile_thread.started.connect(self.profile_worker.run)
        self.profile_worker.finished.connect(self._on_task_card_model_test_finished)
        self.profile_worker.failed.connect(self._on_task_card_model_test_failed)
        self.profile_worker.finished.connect(self.profile_thread.quit); self.profile_worker.failed.connect(self.profile_thread.quit)
        self.profile_thread.finished.connect(self._cleanup_task_card_model_test)
        self.task_model_test_button.setEnabled(False)
        self.task_model_raw_text.clear(); self.task_model_formal_text.clear()
        self.task_model_test_image.clear(); self.task_model_test_image.setText("正在拍摄任务卡并等待大模型返回……")
        self.task_model_test_status.setText("正在拍摄本次新图、发送Qwen并执行正式协议校验……")
        self.task_model_test_status.setStyleSheet("padding:8px;color:#7a5200;background:#fff2bf;font-weight:700;")
        self.profile_thread.start(); self._refresh_authorization_button(); self._refresh_direct_assembly_controls()

    def _on_task_card_model_test_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        raw = value.get("raw_response")
        self.task_model_raw_text.setPlainText(str(raw) if isinstance(raw, str) and raw else "（模型没有返回可显示的原始文本）")
        formal_view = {
            "model_result": value.get("model_result"),
            "formal_result": value.get("formal_result"),
            "validated_result": value.get("validated_result"),
            "validation_message": value.get("validation_message"),
        }
        self.task_model_formal_text.setPlainText(json.dumps(formal_view, ensure_ascii=False, indent=2))
        path = resolve_project_path(str(value.get("image_path", "")))
        if path.is_file():
            pixmap = QPixmap(str(path))
            self.task_model_test_image.setPixmap(pixmap.scaled(760, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.task_model_test_image.setToolTip(str(path))
        protocol_ok = value.get("success") is True
        recognition_ok = value.get("recognition_success") is True
        message = (
            f"帧请求={value.get('request_id')}；拍摄时间={value.get('captured_at')}；"
            f"模型={value.get('model')}；模型请求={value.get('provider_request_id')}；耗时={value.get('elapsed_ms')} ms。\n"
            f"{value.get('validation_message', '')}"
        )
        if protocol_ok and recognition_ok:
            style = "padding:8px;color:#087a26;background:#d7f5dc;font-weight:700;"
        elif protocol_ok:
            style = "padding:8px;color:#7a5200;background:#fff2bf;font-weight:700;"
        else:
            style = "padding:8px;color:#a00000;background:#ffd3d3;font-weight:700;"
        self.task_model_test_status.setText(message); self.task_model_test_status.setStyleSheet(style)
        self._log(f"[task_card_model_test] {message}")

    def _on_task_card_model_test_failed(self, reason: str) -> None:
        message = f"任务卡拍照或大模型测试未完成：{reason}"
        self.task_model_test_status.setText(message)
        self.task_model_test_status.setStyleSheet("padding:8px;color:#a00000;background:#ffd3d3;font-weight:700;")
        self._log(f"[task_card_model_test_failed] {reason}")

    def _cleanup_task_card_model_test(self) -> None:
        thread, worker = self.profile_thread, self.profile_worker
        self.profile_thread = None; self.profile_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.task_model_test_button.setEnabled(True)
        self._refresh_authorization_button(); self._refresh_direct_assembly_controls()

    def _monitor_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        self.monitor_status = QLabel("未启动")
        self.monitor_status.setStyleSheet("font-size:16px;font-weight:700;padding:8px;")
        visual_group = QGroupBox("最近一次颜色识别结果")
        visual_layout = QVBoxLayout(visual_group)
        self.monitor_visual_status = QLabel("尚未拍摄颜色识别图片")
        self.monitor_visual_status.setWordWrap(True)
        self.monitor_visual_image = QLabel("拍照后在这里显示带框结果图")
        self.monitor_visual_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.monitor_visual_image.setMinimumHeight(220)
        self.monitor_visual_image.setStyleSheet("background:#222;color:#ddd;padding:8px;")
        visual_layout.addWidget(self.monitor_visual_status); visual_layout.addWidget(self.monitor_visual_image)
        self.monitor_log = QPlainTextEdit(); self.monitor_log.setReadOnly(True); self.monitor_log.setMaximumBlockCount(200)
        self.monitor_log.setStyleSheet("background:#111;color:#ddd;font-family:Consolas,monospace;")
        layout.addWidget(self.monitor_status); layout.addWidget(visual_group); layout.addWidget(self.monitor_log)
        return page

    def _direct_assembly_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        warning = QLabel(
            "该入口跳过任务卡、语音、大模型和六组编排，只执行一组真实抓取→放置。\n"
            "仍强制检查机器人/相机身份、真实画面目标唯一性、两套正式九点、活动TCP、两个完整红色抓放基准、IO回读和配置指纹；程序不提供碰撞规划。"
        )
        warning.setWordWrap(True); warning.setStyleSheet("padding:12px;background:#fff4cc;font-weight:700;")
        layout.addWidget(warning)

        controls = QGroupBox("单组颜色选择")
        form = QGridLayout(controls)
        self.direct_block_color_combo = QComboBox(); self.direct_tray_color_combo = QComboBox()
        for color in COLORS:
            self.direct_block_color_combo.addItem(color, color); self.direct_tray_color_combo.addItem(color, color)
        form.addWidget(QLabel("抓取方块颜色"), 0, 0); form.addWidget(self.direct_block_color_combo, 0, 1)
        form.addWidget(QLabel("放置托盘颜色"), 0, 2); form.addWidget(self.direct_tray_color_combo, 0, 3)
        self.direct_safety_check = QCheckBox("我确认当前姿态到待机点/方块拍照点/托盘拍照点的直接路径无人且无障碍物，六色托盘均在视野内")
        self.direct_safety_check.stateChanged.connect(self._refresh_direct_assembly_controls)
        form.addWidget(self.direct_safety_check, 1, 0, 1, 4)
        self.start_direct_assembly_button = QPushButton("开始一组真实抓取并放置（跳过任务卡）")
        self.start_direct_assembly_button.setMinimumHeight(48)
        self.start_direct_assembly_button.setStyleSheet("font-weight:700;padding:9px;background:#8a3b00;color:white;")
        self.start_direct_assembly_button.clicked.connect(self._start_direct_assembly)
        self.stop_direct_assembly_button = QPushButton("人工停止单组抓放")
        self.stop_direct_assembly_button.clicked.connect(self._stop_direct_assembly)
        form.addWidget(self.start_direct_assembly_button, 2, 0, 1, 3); form.addWidget(self.stop_direct_assembly_button, 2, 3)
        layout.addWidget(controls)

        self.direct_assembly_status = QLabel("未启动；确认两套九点、真实位姿、颜色参数和现场路径安全后使用。")
        self.direct_assembly_status.setWordWrap(True); self.direct_assembly_status.setStyleSheet("padding:8px;font-weight:700;")
        layout.addWidget(self.direct_assembly_status)
        self.direct_visual_status = QLabel("尚未拍摄单组视觉图片"); self.direct_visual_status.setWordWrap(True)
        self.direct_visual_image = QLabel("运行后显示最近一次带框识别结果")
        self.direct_visual_image.setAlignment(Qt.AlignmentFlag.AlignCenter); self.direct_visual_image.setMinimumHeight(220)
        self.direct_visual_image.setStyleSheet("background:#222;color:#ddd;padding:8px;")
        layout.addWidget(self.direct_visual_status); layout.addWidget(self.direct_visual_image)
        self.direct_assembly_log = QPlainTextEdit(); self.direct_assembly_log.setReadOnly(True); self.direct_assembly_log.setMaximumBlockCount(200)
        self.direct_assembly_log.setStyleSheet("background:#111;color:#ddd;font-family:Consolas,monospace;")
        layout.addWidget(self.direct_assembly_log)
        self._refresh_direct_assembly_controls()
        return page

    @staticmethod
    def _information_page(text: str) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        label = QLabel(text); label.setWordWrap(True); label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setStyleSheet("font-size:15px;padding:18px;")
        layout.addWidget(label); layout.addStretch(); return page

    def _robot_config_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        points_group = QGroupBox("四个固定关节点（运行时使用 moveJoint）")
        points_layout = QVBoxLayout(points_group)
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("关节输入/显示单位"))
        self.joint_unit_combo = QComboBox()
        self.joint_unit_combo.addItem("rad（配置存储单位）", "rad")
        self.joint_unit_combo.addItem("deg（ARCS常见显示）", "deg")
        self._joint_unit_key = "rad"
        self.joint_unit_combo.currentIndexChanged.connect(self._on_joint_unit_changed)
        unit_row.addWidget(self.joint_unit_combo); unit_row.addStretch()
        points_layout.addLayout(unit_row)
        self.points_table = QTableWidget(len(POINT_KEYS), 8)
        self.points_table.setHorizontalHeaderLabels(("点位名称", "J1", "J2", "J3", "J4", "J5", "J6", "操作"))
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.points_table.setMinimumHeight(220)
        self.points_table.setMaximumHeight(260)
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.move_point_buttons: dict[str, QPushButton] = {}
        for row, key in enumerate(POINT_KEYS):
            label = QTableWidgetItem(POINT_LABELS[key]); label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            label.setData(Qt.ItemDataRole.UserRole, key); self.points_table.setItem(row, 0, label)
            for column in range(1, 7): self.points_table.setItem(row, column, QTableWidgetItem(""))
            move_button = QPushButton("移动到此点")
            move_button.clicked.connect(lambda _checked=False, point_key=key: self._start_point_move(point_key))
            self.points_table.setCellWidget(row, 7, move_button)
            self.move_point_buttons[key] = move_button
        self.points_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        points_layout.addWidget(self.points_table)
        self.capture_point_button = QPushButton("读取当前机械臂关节角并直接保存到选中点位（不运动）")
        self.capture_point_button.clicked.connect(self._start_point_capture)
        points_layout.addWidget(self.capture_point_button)
        self.points_table.setCurrentCell(0, 0)
        layout.addWidget(points_group)

        reference_group = QGroupBox("六色独立抓取/放置基准点（现场示教）")
        reference_layout = QGridLayout(reference_group)
        reference_note = QLabel(
            "先完成对应工作区九点并采集六色参考图；从此刻起保持六个目标不动，"
            "再依次对准同色目标并保存完整TCP。重新九点会保留已保存基准，但必须确保六色目标全程未移动；"
            "更改拍照点或活动TCP仍会使对应基准失效。下列按钮只读TCP，不会运动。"
        )
        reference_note.setWordWrap(True); reference_layout.addWidget(reference_note, 0, 0, 1, 3)
        self.reference_anchor_labels: dict[tuple[str, str], QLabel] = {}
        self.reference_anchor_buttons: dict[tuple[str, str], QPushButton] = {}
        row = 1
        for scene, scene_label, action in (("blocks", "Block", "抓取"), ("trays", "Tray", "放置")):
            for color in COLORS:
                label = QLabel(f"{scene_label}/{color}色{action}TCP：未设置")
                button = QPushButton(f"单击立即读取并保存当前TCP为 {scene_label}/{color}色{action}基准")
                button.clicked.connect(lambda _checked=False, s=scene, c=color: self._start_reference_anchor_capture(s, c))
                self.reference_anchor_labels[(scene, color)] = label
                self.reference_anchor_buttons[(scene, color)] = button
                reference_layout.addWidget(label, row, 0, 1, 2); reference_layout.addWidget(button, row, 2)
                row += 1
        # Stable aliases retained for existing UI integrations; these are the red buttons/labels.
        self.blocks_reference_label = self.reference_anchor_labels[("blocks", "红")]
        self.trays_reference_label = self.reference_anchor_labels[("trays", "红")]
        self.capture_blocks_reference_button = self.reference_anchor_buttons[("blocks", "红")]
        self.capture_trays_reference_button = self.reference_anchor_buttons[("trays", "红")]
        layout.addWidget(reference_group)

        controls = QHBoxLayout()
        self.robot_config_confirm = QCheckBox("我确认以上数据抄自当前真实 ARCS，而不是旧仿真或猜测值")
        controls.addWidget(self.robot_config_confirm, 1)
        reload_button = QPushButton("重新加载配置"); reload_button.clicked.connect(self._reload_robot_config)
        save_button = QPushButton("保存四个固定关节点（不连接硬件）"); save_button.clicked.connect(self._save_robot_config)
        controls.addWidget(reload_button); controls.addWidget(save_button)
        layout.addLayout(controls)
        self.robot_config_status = QLabel("尚未加载")
        self.robot_config_status.setStyleSheet("padding:7px;color:#555;"); layout.addWidget(self.robot_config_status)
        self._reload_robot_config()
        return page

    def _camera_config_page(self) -> QWidget:
        page = QWidget(); outer = QVBoxLayout(page); scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content); outer.addWidget(scroll)
        warning = QLabel(
            "按现场决定，勾选确认后可直接保存并人工批准三套采集参数；保存和批准均不连接相机。\n"
            "运行时由MVS SDK只按保存值写入，不做参数回读一致性检查；颜色检测和九点标定仍按各自流程验收。"
        )
        warning.setWordWrap(True); warning.setStyleSheet("padding:10px;background:#fff4cc;font-weight:600;")
        layout.addWidget(warning)

        mounting = QLabel("相机由程序自动枚举并打开；安装方式：眼在手上（固定）")
        mounting.setStyleSheet("padding:7px;color:#555;")
        layout.addWidget(mounting)

        profiles_group = QGroupBox("三套锁定采集参数（全部采用软件触发）")
        profiles_layout = QVBoxLayout(profiles_group)
        profile_tools = QHBoxLayout()
        note = QLabel(
            "曝光单位 us；宽/高/Offset单位 pixel；白平衡填写 MVS中 Red / Green / Blue 的 BalanceRatio。\n"
            "请直接填写已确认的参数；保存并人工批准不连接相机，也不做MVS参数回读。"
        )
        note.setWordWrap(True); profile_tools.addWidget(note, 1)
        self.read_mvs_parameters_button = QPushButton("从当前 MVS 读取并填入上表")
        self.read_mvs_parameters_button.setMinimumHeight(44)
        self.read_mvs_parameters_button.setStyleSheet("font-weight:700;padding:8px 14px;")
        self.read_mvs_parameters_button.setToolTip("只读曝光、增益、白平衡和ROI；不采图、不保存、不批准")
        self.read_mvs_parameters_button.clicked.connect(self._start_current_mvs_read)
        self.read_mvs_parameters_button.setVisible(False)
        profile_tools.addWidget(self.read_mvs_parameters_button)
        profiles_layout.addLayout(profile_tools)
        self.camera_profiles_table = QTableWidget(len(PROFILE_KEYS), 10)
        self.camera_profiles_table.setHorizontalHeaderLabels(("场景", "曝光(us)", "增益", "白平衡R", "白平衡G", "白平衡B", "宽", "高", "OffsetX", "OffsetY"))
        self.camera_profiles_table.verticalHeader().setVisible(False)
        self.camera_profiles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.camera_profiles_table.setMaximumHeight(155)
        for row, key in enumerate(PROFILE_KEYS):
            label = QTableWidgetItem(PROFILE_LABELS[key]); label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            label.setData(Qt.ItemDataRole.UserRole, key); self.camera_profiles_table.setItem(row, 0, label)
            for column in range(1, 10): self.camera_profiles_table.setItem(row, column, QTableWidgetItem(""))
        profiles_layout.addWidget(self.camera_profiles_table); layout.addWidget(profiles_group)

        detector_group = QGroupBox("blocks/trays颜色检测参数（保存后重启MVS服务，再用真实六色同帧验证）")
        detector_layout = QGridLayout(detector_group)
        self.detector_scene_combo = QComboBox(); self.detector_scene_combo.addItem("方块 blocks", "blocks"); self.detector_scene_combo.addItem("托盘 trays", "trays")
        self.detector_scene_combo.currentIndexChanged.connect(self._load_detector_fields)
        detector_layout.addWidget(QLabel("场景"), 0, 0); detector_layout.addWidget(self.detector_scene_combo, 0, 1)
        self.detector_roi_edits = [QLineEdit() for _ in range(4)]
        for column, (label, edit) in enumerate(zip(("x1", "y1", "x2", "y2"), self.detector_roi_edits), start=2):
            edit.setReadOnly(True)
            edit.setToolTip("自动使用本场景完整采集画面，无需填写")
            detector_layout.addWidget(QLabel(label), 0, column * 2 - 2); detector_layout.addWidget(edit, 0, column * 2 - 1)
        self.detector_min_area_edit, self.detector_max_area_edit, self.detector_confidence_edit = QLineEdit(), QLineEdit(), QLineEdit()
        detector_layout.addWidget(QLabel("最小面积px"), 1, 0); detector_layout.addWidget(self.detector_min_area_edit, 1, 1)
        detector_layout.addWidget(QLabel("最大面积px"), 1, 2); detector_layout.addWidget(self.detector_max_area_edit, 1, 3)
        detector_layout.addWidget(QLabel("置信度0..1"), 1, 4); detector_layout.addWidget(self.detector_confidence_edit, 1, 5)
        self.detector_hsv_edit = QPlainTextEdit(); self.detector_hsv_edit.setMaximumHeight(88); detector_layout.addWidget(QLabel("六色HSV JSON"), 2, 0); detector_layout.addWidget(self.detector_hsv_edit, 2, 1, 1, 7)
        self.save_detector_button = QPushButton("保存当前场景面积/HSV到 camera.json（自动填写后点这里）")
        self.save_detector_button.setMinimumHeight(44)
        self.save_detector_button.setStyleSheet("font-weight:700;padding:8px 14px;background:#176b3a;color:white;")
        self.save_detector_button.setToolTip("把当前界面中的最小面积、最大面积、置信度和六色HSV写入 config/real/camera.json；保存会撤销旧的六色批准")
        self.save_detector_button.clicked.connect(self._save_detector_fields)
        self.estimate_detector_area_button = QPushButton("真实拍照并自动填写面积"); self.estimate_detector_area_button.clicked.connect(self._start_detector_area_estimate)
        self.validate_detector_button = QPushButton("真实新帧验证六色并批准"); self.validate_detector_button.clicked.connect(self._start_detector_validation)
        detector_layout.addWidget(self.estimate_detector_area_button, 3, 0, 1, 5)
        detector_layout.addWidget(self.validate_detector_button, 3, 5, 1, 5)
        detector_layout.addWidget(self.save_detector_button, 4, 0, 1, 10)
        self.detector_editor_group = detector_group
        detector_group.setVisible(False)
        layout.addWidget(detector_group)

        detector_workflow = QGroupBox("颜色参数处理方式")
        detector_workflow_layout = QVBoxLayout(detector_workflow)
        self.detector_workflow_note = QLabel(
            "面积、置信度和六色HSV不在本界面手工编辑。请使用下方 Block/Tray 拍照取图，"
            "再根据当前现场实拍图片离线分析并直接写入 config/real/camera.json；写入后重启MVS服务。"
            "“拍照并用当前参数识别”会对一张全新帧使用已保存参数诊断，不保存或批准配置、不运动机器人。"
        )
        self.detector_workflow_note.setWordWrap(True)
        self.detector_workflow_note.setStyleSheet("padding:10px;background:#e8f1ff;color:#17365d;font-weight:700;")
        detector_workflow_layout.addWidget(self.detector_workflow_note)
        layout.addWidget(detector_workflow)

        visual_group = QGroupBox("最近一次颜色识别结果（绿色=选中，黄色=候选冲突，红色=被过滤）")
        visual_layout = QVBoxLayout(visual_group)
        manual_capture_layout = QHBoxLayout()
        self.manual_block_capture_button = QPushButton("Block 拍照取图")
        self.manual_tray_capture_button = QPushButton("Tray 拍照取图")
        self.manual_block_recognize_button = QPushButton("Block 拍照并用当前参数识别")
        self.manual_tray_recognize_button = QPushButton("Tray 拍照并用当前参数识别")
        self.manual_block_capture_button.setToolTip("使用 blocks 采集参数拍摄并保存一张原图；不做识别、不批准参数")
        self.manual_tray_capture_button.setToolTip("使用 trays 采集参数拍摄并保存一张原图；不做识别、不批准参数")
        self.manual_block_recognize_button.setToolTip("新拍一帧并使用已保存 camera.json 与可用九点标定识别；不运动、不批准、不改配置")
        self.manual_tray_recognize_button.setToolTip("新拍一帧并使用已保存 camera.json 与可用九点标定识别；不运动、不批准、不改配置")
        self.manual_block_capture_button.clicked.connect(lambda: self._start_manual_scene_capture("blocks"))
        self.manual_tray_capture_button.clicked.connect(lambda: self._start_manual_scene_capture("trays"))
        self.manual_block_recognize_button.clicked.connect(lambda: self._start_manual_scene_recognition("blocks"))
        self.manual_tray_recognize_button.clicked.connect(lambda: self._start_manual_scene_recognition("trays"))
        manual_capture_layout.addWidget(self.manual_block_capture_button)
        manual_capture_layout.addWidget(self.manual_tray_capture_button)
        manual_capture_layout.addWidget(self.manual_block_recognize_button)
        manual_capture_layout.addWidget(self.manual_tray_recognize_button)
        visual_layout.addLayout(manual_capture_layout)
        self.detector_visual_status = QLabel("尚未拍摄颜色识别图片")
        self.detector_visual_status.setWordWrap(True)
        self.detector_visual_image = QLabel("拍照或运行识别后在这里显示原图/识别框")
        self.detector_visual_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detector_visual_image.setMinimumHeight(220)
        self.detector_visual_image.setStyleSheet("background:#222;color:#ddd;padding:8px;")
        visual_layout.addWidget(self.detector_visual_status); visual_layout.addWidget(self.detector_visual_image)
        layout.addWidget(visual_group)

        evidence = QGroupBox("运行时自动产生/其他页面提供（不要手填）")
        evidence_layout = QVBoxLayout(evidence)
        self.camera_evidence_label = QLabel(
            "• 当前帧编号、拍摄时间：每次真实软件触发后由 MVS SDK自动记录。\n"
            "• 三套参数批准：由现场操作者确认；运行时只按保存值写入，不做MVS参数回读。\n"
            "• 拍照点、活动 TCP：读取“机器人现场配置”页。\n"
            "• blocks / trays 标定 ID：完成各自真实九点标定后生成；任务卡不需要九点标定。\n"
            "• 检测 ROI：程序固定为本场景完整采集画面 [0, 0, Width, Height]，无需手填。\n"
            "• HSV和面积阈值：根据本机当前现场实拍图离线分析后直接写入 camera.json。"
        )
        self.camera_evidence_label.setWordWrap(True); evidence_layout.addWidget(self.camera_evidence_label); layout.addWidget(evidence)

        controls = QHBoxLayout()
        self.camera_config_confirm = QCheckBox("我确认三套数值来自当前真实 MVS相机，不是旧图片或猜测值")
        controls.addWidget(self.camera_config_confirm, 1)
        reload_button = QPushButton("重新加载配置"); reload_button.clicked.connect(self._reload_camera_config)
        save_button = QPushButton("保存并人工批准 MVS参数（不连接相机）"); save_button.clicked.connect(self._save_camera_config)
        self.validate_profiles_button = QPushButton("人工确认批准已保存三套参数（不回读）"); self.validate_profiles_button.clicked.connect(self._approve_profiles_without_readback)
        controls.addWidget(reload_button); controls.addWidget(save_button); controls.addWidget(self.validate_profiles_button); layout.addLayout(controls)
        self.camera_config_status = QLabel("尚未加载"); self.camera_config_status.setWordWrap(True); self.camera_config_status.setStyleSheet("padding:7px;color:#555;")
        layout.addWidget(self.camera_config_status); layout.addStretch()
        self._reload_camera_config()
        self._load_detector_fields()
        return page

    def _manual_io_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        warning = QLabel(
            "本页按钮会连接 config/real 中指定的唯一 AUBO，并真实写入标准数字输出。\n"
            "手动IO不受相机、九点、点位、TCP、Runtime和批准基线影响；仍核对唯一机器人身份、RobotMode=Running、SafetyMode=Normal和空执行队列。\n"
            "已确认映射：标准DO0控制光圈；吸盘使用工具IO，TOOL_IO[1]长期保持1，TOOL_IO[0]为0吸取、1不吸。"
        )
        warning.setWordWrap(True); warning.setStyleSheet("padding:12px;background:#ffd9a8;font-weight:700;")
        layout.addWidget(warning)
        group = QGroupBox("真实 IO 单步操作")
        controls = QGridLayout(group)
        self.aperture_toggle_button = QPushButton("切换光圈 DO0（先读取，再切换）")
        self.aperture_toggle_button.clicked.connect(lambda: self._start_manual_io("toggle_aperture"))
        self.suction_on_button = QPushButton("吸盘吸取：TOOL_IO[1]=1，TOOL_IO[0]=0")
        self.suction_on_button.clicked.connect(lambda: self._start_manual_io("suction_on"))
        self.suction_off_button = QPushButton("吸盘不吸：保持TOOL_IO[1]=1，TOOL_IO[0]=1")
        self.suction_off_button.clicked.connect(lambda: self._start_manual_io("suction_off"))
        controls.addWidget(self.aperture_toggle_button, 0, 0, 1, 2)
        controls.addWidget(self.suction_on_button, 1, 0)
        controls.addWidget(self.suction_off_button, 1, 1)
        layout.addWidget(group)
        self.io_status = QLabel("尚未执行；界面不假定当前真实 IO 状态。")
        self.io_status.setWordWrap(True); self.io_status.setStyleSheet("padding:10px;color:#555;")
        layout.addWidget(self.io_status); layout.addStretch()
        return page

    def _reload_camera_config(self) -> None:
        try:
            profiles = load_camera_editor_values(REAL_CONFIG_DIR / "camera.json")
        except CameraConfigInputError as exc:
            self.camera_config_status.setText(f"加载失败：{exc}"); self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        fields = ("exposure_us", "gain", "white_red", "white_green", "white_blue", "width", "height", "offset_x", "offset_y")
        for row, key in enumerate(PROFILE_KEYS):
            for column, field in enumerate(fields, start=1):
                value = profiles[key][field]
                self.camera_profiles_table.item(row, column).setText("" if value is None else f"{value:.12g}")
        self.camera_config_confirm.setChecked(False)
        self.camera_config_status.setText("已从 config/real/camera.json加载；空白表示尚未配置。")
        self.camera_config_status.setStyleSheet("padding:7px;color:#087a26;")

    def _save_camera_config(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.mvs_read_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            self.camera_config_status.setText("比赛、九点标定或真实视觉验证运行中，禁止修改相机配置。")
            self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        if not self.camera_config_confirm.isChecked():
            self.camera_config_status.setText("请先勾选当前真实 MVS数据确认。")
            self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        fields = ("exposure_us", "gain", "white_red", "white_green", "white_blue", "width", "height", "offset_x", "offset_y")
        values = {
            key: {field: self.camera_profiles_table.item(row, column).text().strip() for column, field in enumerate(fields, start=1)}
            for row, key in enumerate(PROFILE_KEYS)
        }
        try:
            save_camera_editor_values(REAL_CONFIG_DIR / "camera.json", profile_values=values)
            approve_profile_batch_by_operator(REAL_CONFIG_DIR / "camera.json")
        except (CameraConfigInputError, OSError) as exc:
            self.camera_config_status.setText(f"保存失败：{exc}"); self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        if self.session.state.value != "idle":
            self.session.revoke("MVS现场配置已修改")
            self.session_status.setText("授权已撤销：MVS现场配置已修改")
        self.camera_config_confirm.setChecked(False)
        self.camera_config_status.setText("保存成功；三套采集参数已按现场确认人工批准，不连接相机、不做预先回读。颜色检测仍需单独处理。")
        self.camera_config_status.setStyleSheet("padding:7px;color:#087a26;")
        self._log("用户保存三套 MVS采集参数并人工批准；未枚举、打开、回读或触发相机。")
        self.run_preflight()

    def _approve_profiles_without_readback(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.mvs_read_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        if not self.camera_config_confirm.isChecked():
            self.camera_config_status.setText("请先勾选当前真实 MVS数据确认。")
            self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        try:
            approve_profile_batch_by_operator(REAL_CONFIG_DIR / "camera.json")
        except (CameraConfigInputError, OSError) as exc:
            self.camera_config_status.setText(f"人工批准失败：{exc}")
            self.camera_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        self.camera_config_confirm.setChecked(False)
        self.camera_config_status.setText("三套采集参数已按现场确认人工批准；未连接相机、未回读、未触发。")
        self.camera_config_status.setStyleSheet("padding:7px;color:#087a26;")
        self._log("用户人工批准已保存的三套MVS采集参数；未连接相机或回读。")
        self.run_preflight()

    def _start_current_mvs_read(self) -> None:
        if any(thread is not None for thread in (
            self.mvs_read_thread, self.profile_thread, self.detector_thread, self.competition_thread,
            self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread,
            self.point_capture_thread, self.point_move_thread, self.contact_capture_thread,
        )):
            return
        answer = QMessageBox.question(
            self,
            "确认只读真实MVS参数",
            "将自动枚举并独占连接第一台可用 MVS相机，只读取当前曝光、增益、白平衡R/G/B、Width/Height/Offset。\n\n"
            "不会启动采集、不会软件触发、不会修改参数数值；读取白平衡时会切换R/G/B查看选择器。"
            "请先完全关闭MVS客户端和真实MVS视觉服务，避免相机被占用。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.mvs_read_thread = QThread(self); self.mvs_read_worker = CurrentMvsParametersWorker()
        self.mvs_read_worker.moveToThread(self.mvs_read_thread); self.mvs_read_thread.started.connect(self.mvs_read_worker.run)
        self.mvs_read_worker.finished.connect(self._on_current_mvs_read); self.mvs_read_worker.failed.connect(self._on_current_mvs_read_failed)
        self.mvs_read_worker.finished.connect(self.mvs_read_thread.quit); self.mvs_read_worker.failed.connect(self.mvs_read_thread.quit)
        self.mvs_read_thread.finished.connect(self._cleanup_current_mvs_read)
        self.read_mvs_parameters_button.setEnabled(False)
        self.camera_config_status.setText("正在自动连接可用 MVS相机；不采图、不触发、不修改参数数值……")
        self.mvs_read_thread.start(); self._refresh_authorization_button()

    def _on_current_mvs_read(self, payload: object) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
            self.camera_config_status.setText("MVS只读结果格式无效。"); return
        parameters = payload["parameters"]
        try:
            white = parameters["white_balance"]; roi = parameters["roi"]
            values = (
                parameters["exposure_us"], parameters["gain"], white["red"], white["green"], white["blue"],
                roi["width"], roi["height"], roi["offset_x"], roi["offset_y"],
            )
        except (KeyError, TypeError) as exc:
            self.camera_config_status.setText(f"MVS只读字段缺失：{exc}"); return
        for row in range(len(PROFILE_KEYS)):
            for column, value in enumerate(values, start=1):
                self.camera_profiles_table.item(row, column).setText(str(value))
        self.camera_config_status.setText(
            f"已从 {payload.get('model')} 只读当前参数并填入三套表格；"
            "尚未保存、尚未触发拍照、尚未批准。请检查后勾选确认并保存。"
        )

    def _on_current_mvs_read_failed(self, reason: str) -> None:
        self.camera_config_status.setText(f"MVS当前参数只读失败，表格未修改：{reason}")

    def _cleanup_current_mvs_read(self) -> None:
        thread, worker = self.mvs_read_thread, self.mvs_read_worker
        self.mvs_read_thread = None; self.mvs_read_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.read_mvs_parameters_button.setEnabled(True); self._refresh_authorization_button()

    def _start_profile_validation(self) -> None:
        if any(thread is not None for thread in (self.profile_thread, self.mvs_read_thread, self.detector_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        self.profile_thread = QThread(self); self.profile_worker = ProfileValidationWorker()
        self.profile_worker.moveToThread(self.profile_thread); self.profile_thread.started.connect(self.profile_worker.run)
        self.profile_worker.finished.connect(self._on_profile_validation); self.profile_worker.failed.connect(self._on_profile_validation_failed)
        self.profile_worker.finished.connect(self.profile_thread.quit); self.profile_worker.failed.connect(self.profile_thread.quit)
        self.profile_thread.finished.connect(self._cleanup_profile_worker)
        self.validate_profiles_button.setEnabled(False)
        self.camera_config_status.setText("正在使用自动枚举的真实MVS，依次设置、回读并触发 task_card / blocks / trays……")
        self.profile_thread.start(); self._refresh_authorization_button()

    def _on_profile_validation(self, payload: object) -> None:
        if not isinstance(payload, list) or len(payload) != 3:
            self.camera_config_status.setText("三套采集参数写入测试结果数量无效。"); return
        try:
            if not all(isinstance(result, dict) for result in payload): raise CameraConfigInputError("采集参数写入测试结果不是对象。")
            if {str(result.get("profile")) for result in payload} != set(PROFILE_KEYS): raise CameraConfigInputError("采集参数写入测试未唯一覆盖三套 profile。")
            approve_profile_batch(
                REAL_CONFIG_DIR / "camera.json",
                expected_sha256_by_profile={str(result["profile"]): str(result["profile_sha256"]) for result in payload},
            )
        except Exception as exc:
            self.camera_config_status.setText(f"参数写入测试完成但批准失败：{exc}"); return
        summary = "；".join(
            f"{item['profile']}: 帧{item.get('frame_number')} {item.get('captured_at')} {item.get('image_width')}×{item.get('image_height')} 写入值={json.dumps(item.get('configured_parameters'), ensure_ascii=False)}"
            for item in payload if isinstance(item, dict)
        )
        self._reload_camera_config(); self.run_preflight()
        self.camera_config_status.setText(f"三套参数写入和取图测试成功，已批准。请重启MVS服务后再做六色验证。{summary}")

    def _on_profile_validation_failed(self, reason: str) -> None:
        self.camera_config_status.setText(f"三套参数写入测试失败，未批准：{reason}")

    def _cleanup_profile_worker(self) -> None:
        thread, worker = self.profile_thread, self.profile_worker
        self.profile_thread = None; self.profile_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.validate_profiles_button.setEnabled(True); self._refresh_authorization_button()

    def _load_detector_fields(self, _index: int = 0) -> None:
        try:
            scene = str(self.detector_scene_combo.currentData()); detector = load_detector_editor_values(REAL_CONFIG_DIR / "camera.json")[scene]
        except (CameraConfigInputError, KeyError) as exc:
            self.camera_config_status.setText(f"颜色参数加载失败：{exc}"); return
        roi = detector.get("roi") if isinstance(detector.get("roi"), list) else [None] * 4
        for edit, value in zip(self.detector_roi_edits, roi): edit.setText("" if value is None else str(value))
        self.detector_min_area_edit.setText("" if detector.get("min_area_px") == "UNSET" else str(detector.get("min_area_px", "")))
        self.detector_max_area_edit.setText("" if detector.get("max_area_px") == "UNSET" else str(detector.get("max_area_px", "")))
        self.detector_confidence_edit.setText(str(detector.get("confidence_min", 0.6)))
        self.detector_hsv_edit.setPlainText(json.dumps(detector.get("hsv_ranges", {}), ensure_ascii=False, indent=2))
        self.camera_config_status.setText(f"已加载 {scene}颜色参数；检测ROI自动使用完整画面；approved={detector.get('approved')}")

    def _save_detector_fields(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.mvs_read_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            self.camera_config_status.setText("比赛、九点标定或真实视觉验证运行中，禁止修改颜色参数。"); return
        scene = str(self.detector_scene_combo.currentData())
        try:
            changed = save_detector_editor_values(
                REAL_CONFIG_DIR / "camera.json", scene=scene,
                roi_values=[edit.text() for edit in self.detector_roi_edits],
                confidence_min=self.detector_confidence_edit.text(), min_area_px=self.detector_min_area_edit.text(),
                max_area_px=self.detector_max_area_edit.text(), hsv_json=self.detector_hsv_edit.toPlainText(),
            )
        except (CameraConfigInputError, OSError) as exc:
            self.camera_config_status.setText(f"颜色参数保存失败：{exc}"); return
        self.run_preflight(); self._load_detector_fields()
        message = (
            f"{scene}面积/HSV已写入且参数有变化，旧批准已撤销；请重启MVS服务，再把真实六色同时放入画面验证。"
            if changed else
            f"{scene}面积/HSV与已保存值完全相同；已保留当前六色批准，无需重复验证。"
        )
        self.camera_config_status.setText(message); self._log(message)

    def _start_detector_validation(self) -> None:
        if any(thread is not None for thread in (self.detector_thread, self.profile_thread, self.mvs_read_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        scene = str(self.detector_scene_combo.currentData())
        self.detector_thread = QThread(self); self.detector_worker = DetectorValidationWorker(scene=scene)
        self.detector_worker.moveToThread(self.detector_thread); self.detector_thread.started.connect(self.detector_worker.run)
        self.detector_worker.visual.connect(self._on_detector_visual_result)
        self.detector_worker.finished.connect(self._on_detector_validation); self.detector_worker.failed.connect(self._on_detector_validation_failed)
        self.detector_worker.finished.connect(self.detector_thread.quit); self.detector_worker.failed.connect(self.detector_thread.quit)
        self.detector_thread.finished.connect(self._cleanup_detector_worker)
        self.validate_detector_button.setEnabled(False); self.estimate_detector_area_button.setEnabled(False)
        self.manual_block_capture_button.setEnabled(False); self.manual_tray_capture_button.setEnabled(False)
        self.manual_block_recognize_button.setEnabled(False); self.manual_tray_recognize_button.setEnabled(False)
        self.camera_config_status.setText("正在触发真实MVS新帧并验证六色唯一性……")
        self.detector_thread.start()

    def _start_detector_area_estimate(self) -> None:
        if any(thread is not None for thread in (self.detector_thread, self.profile_thread, self.mvs_read_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        scene = str(self.detector_scene_combo.currentData())
        answer = QMessageBox.question(
            self, "确认真实拍照估算面积",
            f"将连接真实MVS并对 {scene} 软件触发拍摄一张新图。请确认红、橙、黄、绿、蓝、紫六个真实目标均在画面中，且背景没有相近同色方形物。\n\n"
            "程序只会把估算结果填入最小/最大面积框，不会自动保存或批准。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.detector_thread = QThread(self); self.detector_worker = DetectorAreaEstimateWorker(scene=scene)
        self.detector_worker.moveToThread(self.detector_thread); self.detector_thread.started.connect(self.detector_worker.run)
        self.detector_worker.finished.connect(self._on_detector_area_estimate); self.detector_worker.failed.connect(self._on_detector_area_estimate_failed)
        self.detector_worker.finished.connect(self.detector_thread.quit); self.detector_worker.failed.connect(self.detector_thread.quit)
        self.detector_thread.finished.connect(self._cleanup_detector_worker)
        self.estimate_detector_area_button.setEnabled(False); self.validate_detector_button.setEnabled(False)
        self.manual_block_capture_button.setEnabled(False); self.manual_tray_capture_button.setEnabled(False)
        self.manual_block_recognize_button.setEnabled(False); self.manual_tray_recognize_button.setEnabled(False)
        self.camera_config_status.setText(f"正在触发真实MVS新帧并自动估算 {scene} 六色目标面积……")
        self.detector_thread.start(); self._refresh_authorization_button()

    def _on_detector_area_estimate(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self.camera_config_status.setText("面积自动估算结果无效。"); return
        self.detector_min_area_edit.setText(str(payload.get("min_area_px", "")))
        self.detector_max_area_edit.setText(str(payload.get("max_area_px", "")))
        self.camera_config_status.setText(
            f"已根据真实新帧自动填写面积候选：min={payload.get('min_area_px')}，max={payload.get('max_area_px')}；"
            f"六色实测={json.dumps(payload.get('areas_px'), ensure_ascii=False)}。尚未保存；请点击绿色按钮“保存当前场景面积/HSV到 camera.json”。"
        )

    def _on_detector_area_estimate_failed(self, reason: str) -> None:
        self.camera_config_status.setText(f"真实面积自动估算失败，未修改面积框：{reason}")

    def _on_detector_validation(self, payload: object) -> None:
        if not isinstance(payload, dict): return
        try:
            approve_detector_values(REAL_CONFIG_DIR / "camera.json", scene=str(payload["scene"]), expected_sha256=str(payload["detector_sha256"]))
        except Exception as exc:
            self.camera_config_status.setText(f"六色检测通过但批准失败：{exc}"); return
        self.camera_config_status.setText(f"{payload['scene']}真实六色同帧验证通过并批准；请重启MVS服务使批准状态生效。帧={payload.get('frame_number')} 时间={payload.get('captured_at')}")
        self.run_preflight(); self._load_detector_fields()

    def _on_detector_validation_failed(self, reason: str) -> None:
        self.camera_config_status.setText(f"真实六色验证失败：{reason}")

    def _on_detector_visual_result(self, payload: object) -> None:
        self._show_visual_result(payload, self.detector_visual_status, self.detector_visual_image)

    def _start_manual_scene_capture(self, scene: str) -> None:
        if scene not in {"blocks", "trays"}:
            return
        if any(thread is not None for thread in (self.detector_thread, self.profile_thread, self.mvs_read_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        self.detector_thread = QThread(self); self.detector_worker = ManualSceneCaptureWorker(scene=scene)
        self.detector_worker.moveToThread(self.detector_thread); self.detector_thread.started.connect(self.detector_worker.run)
        self.detector_worker.finished.connect(self._on_manual_scene_capture)
        self.detector_worker.failed.connect(self._on_manual_scene_capture_failed)
        self.detector_worker.finished.connect(self.detector_thread.quit); self.detector_worker.failed.connect(self.detector_thread.quit)
        self.detector_thread.finished.connect(self._cleanup_detector_worker)
        self.validate_detector_button.setEnabled(False); self.estimate_detector_area_button.setEnabled(False)
        self.manual_block_capture_button.setEnabled(False); self.manual_tray_capture_button.setEnabled(False)
        self.manual_block_recognize_button.setEnabled(False); self.manual_tray_recognize_button.setEnabled(False)
        label = "Block" if scene == "blocks" else "Tray"
        self.camera_config_status.setText(f"正在使用 {scene} 采集参数拍摄 {label} 原图……")
        self.detector_thread.start(); self._refresh_authorization_button()

    def _on_manual_scene_capture(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_manual_scene_capture_failed("取图结果格式无效。")
            return
        scene, image_path = str(payload.get("scene", "")), str(payload.get("image_path", ""))
        label = "Block" if scene == "blocks" else "Tray"
        message = f"{label} 取图成功，原图已保存：{image_path}"
        self.camera_config_status.setText(message)
        self._show_visual_result({
            "success": True,
            "message": message,
            "annotated_image_path": image_path,
            "detection_summary": {"success": True, "colors": []},
        }, self.detector_visual_status, self.detector_visual_image)
        self._log(message)

    def _on_manual_scene_capture_failed(self, reason: str) -> None:
        message = f"手动拍照取图失败：{reason}"
        self.camera_config_status.setText(message)
        self.detector_visual_status.setText(message)
        self.detector_visual_status.setStyleSheet("padding:8px;font-weight:700;color:#a00000;background:#ffd3d3;")
        self._log(message)

    def _start_manual_scene_recognition(self, scene: str) -> None:
        if scene not in {"blocks", "trays"}:
            return
        if any(thread is not None for thread in (self.detector_thread, self.profile_thread, self.mvs_read_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        self.detector_thread = QThread(self); self.detector_worker = ManualSceneRecognitionWorker(scene=scene)
        self.detector_worker.moveToThread(self.detector_thread); self.detector_thread.started.connect(self.detector_worker.run)
        self.detector_worker.finished.connect(self._on_manual_scene_recognition)
        self.detector_worker.failed.connect(self._on_manual_scene_recognition_failed)
        self.detector_worker.finished.connect(self.detector_thread.quit); self.detector_worker.failed.connect(self.detector_thread.quit)
        self.detector_thread.finished.connect(self._cleanup_detector_worker)
        self.validate_detector_button.setEnabled(False); self.estimate_detector_area_button.setEnabled(False)
        self.manual_block_capture_button.setEnabled(False); self.manual_tray_capture_button.setEnabled(False)
        self.manual_block_recognize_button.setEnabled(False); self.manual_tray_recognize_button.setEnabled(False)
        label = "Block" if scene == "blocks" else "Tray"
        self.camera_config_status.setText(f"正在新拍一帧 {label} 图像，并使用当前已保存参数识别同一帧……")
        self.detector_thread.start(); self._refresh_authorization_button()

    def _on_manual_scene_recognition(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_manual_scene_recognition_failed("拍照识别结果格式无效。")
            return
        detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
        lines: list[str] = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            color = str(item.get("color", "?"))
            pixel = f"像素=({item.get('current_pixel_u', item.get('pixel_u'))}, {item.get('current_pixel_v', item.get('pixel_v'))})"
            if "delta_x_tool_m" in item:
                angle = math.degrees(float(item.get("delta_r_rad", 0.0)))
                lines.append(
                    f"{color}：{pixel}，相对红色基准X={float(item['dx_tool_m']) * 1000.0:+.3f} mm，"
                    f"Y={float(item['dy_tool_m']) * 1000.0:+.3f} mm；"
                    f"相对本色原点ΔX={float(item['delta_x_tool_m']) * 1000.0:+.3f} mm，"
                    f"ΔY={float(item['delta_y_tool_m']) * 1000.0:+.3f} mm，ΔR={angle:+.2f}°，"
                    f"置信度={item.get('confidence')}"
                )
            else:
                lines.append(
                    f"{color}：{pixel}，图像角度={item.get('r_image_deg')}°，置信度={item.get('confidence')}"
                )
        missing = payload.get("missing_colors") if isinstance(payload.get("missing_colors"), list) else []
        if missing:
            lines.append(f"未识别颜色：{', '.join(str(color) for color in missing)}")
        label = "Block" if payload.get("scene") == "blocks" else "Tray"
        message = (
            f"{label} 新帧识别完成：帧={payload.get('frame_number')}，时间={payload.get('captured_at')}。"
            f"{payload.get('calibration_message', '')}"
            + ("\n" + "\n".join(lines) if lines else "")
        )
        value = dict(payload); value["message"] = message
        self.camera_config_status.setText(message)
        self._show_visual_result(value, self.detector_visual_status, self.detector_visual_image)
        self._log(f"{label} 手动新帧识别完成；识别{len(detections)}色，缺失{missing}。")

    def _on_manual_scene_recognition_failed(self, reason: str) -> None:
        message = f"手动拍照并识别失败：{reason}"
        self.camera_config_status.setText(message)
        self.detector_visual_status.setText(message)
        self.detector_visual_status.setStyleSheet("padding:8px;font-weight:700;color:#a00000;background:#ffd3d3;")
        self._log(message)

    def _cleanup_detector_worker(self) -> None:
        thread, worker = self.detector_thread, self.detector_worker
        self.detector_thread = None; self.detector_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.validate_detector_button.setEnabled(True); self.estimate_detector_area_button.setEnabled(True)
        self.manual_block_capture_button.setEnabled(True); self.manual_tray_capture_button.setEnabled(True)
        self.manual_block_recognize_button.setEnabled(True); self.manual_tray_recognize_button.setEnabled(True)

    def _reload_robot_config(self) -> None:
        try:
            _name, _offset, points = load_robot_editor_values(REAL_CONFIG_DIR / "robot.json", REAL_CONFIG_DIR / "motion.json")
            anchors = load_reference_anchors(REAL_CONFIG_DIR / "motion.json")
        except RobotConfigInputError as exc:
            self.robot_config_status.setText(f"加载失败：{exc}"); self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;")
            return
        self.joint_unit_combo.blockSignals(True); self.joint_unit_combo.setCurrentIndex(0); self.joint_unit_combo.blockSignals(False); self._joint_unit_key = "rad"
        for row, key in enumerate(POINT_KEYS):
            for column, value in enumerate(points[key] or [None] * 6, start=1): self.points_table.item(row, column).setText("" if value is None else f"{value:.12g}")
        for scene, scene_label, action in (("blocks", "Block", "抓取"), ("trays", "Tray", "放置")):
            for color in COLORS:
                pose = anchors[scene][color]
                self.reference_anchor_labels[(scene, color)].setText(
                    f"{scene_label}/{color}色{action}TCP：{pose if pose is not None else '未设置'}"
                )
        self.robot_config_confirm.setChecked(False)
        self.robot_config_status.setText("已从 config/real加载；空白表示尚未配置。")
        self.robot_config_status.setStyleSheet("padding:7px;color:#087a26;")

    def _start_point_capture(self) -> None:
        if self.session.state.value != "idle" or any(thread is not None for thread in (
            self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread,
            self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread,
        )):
            QMessageBox.critical(self, "不能采集点位", "比赛会话或其他维护流程正在运行。")
            return
        row = self.points_table.currentRow()
        if not 0 <= row < len(POINT_KEYS):
            QMessageBox.critical(self, "不能采集点位", "请先在表格中选择一个点位。")
            return
        point_key = POINT_KEYS[row]; point_label = POINT_LABELS[point_key]
        answer = QMessageBox.question(
            self,
            "确认只读采集并保存点位",
            f"将连接唯一机器人rob1，只读取当前J1～J6并直接覆盖保存为“{point_label}”。\n\n"
            "机械臂不会运动、不会上电/startup、不会写IO。请确认当前姿态确实就是该点位，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.point_capture_thread = QThread(self)
        self.point_capture_worker = RobotPointCaptureWorker(point_key)
        self.point_capture_worker.moveToThread(self.point_capture_thread)
        self.point_capture_thread.started.connect(self.point_capture_worker.run)
        self.point_capture_worker.finished.connect(self._on_point_capture_finished)
        self.point_capture_worker.failed.connect(self._on_point_capture_failed)
        self.point_capture_worker.finished.connect(self.point_capture_thread.quit)
        self.point_capture_worker.failed.connect(self.point_capture_thread.quit)
        self.point_capture_thread.finished.connect(self._cleanup_point_capture_worker)
        self.capture_point_button.setEnabled(False)
        self.robot_config_status.setText(f"正在只读采集{point_label}；机械臂不会运动……")
        self.point_capture_thread.start(); self._refresh_authorization_button()

    def _on_point_capture_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        point_key = str(value.get("point_key", "")); row = POINT_KEYS.index(point_key) if point_key in POINT_KEYS else 0
        joints = [float(item) for item in value.get("joint_positions_rad", [])]
        self._reload_robot_config(); self.points_table.setCurrentCell(row, 0)
        degrees = [math.degrees(item) for item in joints]
        message = (
            f"已从机器人{value.get('robot_name')}只读采集并保存{value.get('point_label')}。"
            f" rad={joints}；deg={[round(item, 6) for item in degrees]}。"
        )
        self.robot_config_status.setText(message); self.robot_config_status.setStyleSheet("padding:7px;color:#087a26;font-weight:700;")
        self._log(message); self.run_preflight()

    def _on_point_capture_failed(self, reason: str) -> None:
        self.robot_config_status.setText(f"点位采集失败，未保存：{reason}")
        self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;font-weight:700;")
        self._log(f"真实点位只读采集失败：{reason}")

    def _start_point_move(self, point_key: str) -> None:
        if self.session.state.value != "idle" or any(thread is not None for thread in (
            self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread,
            self.profile_thread, self.mvs_read_thread, self.readiness_thread, self.io_thread,
            self.point_capture_thread, self.point_move_thread, self.contact_capture_thread,
        )):
            QMessageBox.critical(self, "不能移动到点位", "比赛会话或其他维护流程正在运行。")
            return
        if point_key not in POINT_KEYS:
            QMessageBox.critical(self, "不能移动到点位", "目标点位无效。")
            return
        row = POINT_KEYS.index(point_key)
        self.points_table.setCurrentCell(row, 0)
        point_label = POINT_LABELS[point_key]
        try:
            _name, _offset, points = load_robot_editor_values(
                REAL_CONFIG_DIR / "robot.json", REAL_CONFIG_DIR / "motion.json"
            )
            target = points[point_key]
            configs = load_all()
            speed_percent = float(maintenance_point_limits(configs["motion"])["speed_fraction"]) * 100.0
        except Exception as exc:
            QMessageBox.critical(self, "不能移动到点位", f"读取正式点位失败：{exc}")
            return
        if target is None:
            QMessageBox.critical(self, "不能移动到点位", f"{point_label}尚未保存完整六轴关节角。")
            return
        answer = QMessageBox.question(
            self,
            "确认真实机械臂移动",
            f"将连接唯一机器人 rob1，以 {speed_percent:g}% 速度直接 MoveJoint 到“{point_label}”。\n\n"
            f"目标关节角(rad)：{[round(float(value), 6) for value in target]}\n\n"
            "程序不提供碰撞规划。请确认机械臂已在 ARCS 上电并 startup，急停可用，"
            "且当前位置到目标点的整条路径无人、无障碍物。是否开始真实运动？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.point_move_thread = QThread(self)
        self.point_move_worker = RobotPointMoveWorker(point_key)
        self.point_move_worker.moveToThread(self.point_move_thread)
        self.point_move_thread.started.connect(self.point_move_worker.run)
        self.point_move_worker.finished.connect(self._on_point_move_finished)
        self.point_move_worker.failed.connect(self._on_point_move_failed)
        self.point_move_worker.finished.connect(self.point_move_thread.quit)
        self.point_move_worker.failed.connect(self.point_move_thread.quit)
        self.point_move_thread.finished.connect(self._cleanup_point_move_worker)
        self.capture_point_button.setEnabled(False)
        for button in self.move_point_buttons.values():
            button.setEnabled(False)
        self.robot_config_status.setText(f"正在以 {speed_percent:g}% 速度移动到{point_label}；请监看机械臂并保持急停可用……")
        self.robot_config_status.setStyleSheet("padding:7px;color:#a05a00;font-weight:700;")
        self.point_move_thread.start(); self._refresh_authorization_button()

    def _on_point_move_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        point_key = str(value.get("point_key", ""))
        if point_key in POINT_KEYS:
            self.points_table.setCurrentCell(POINT_KEYS.index(point_key), 0)
        message = (
            f"机器人{value.get('robot_name')}已到达{value.get('point_label')}；"
            f"速度比例={float(value.get('speed_fraction', 0.0)) * 100:g}%，"
            f"实际关节角(rad)={value.get('joint_positions_rad')}。"
        )
        self.robot_config_status.setText(message)
        self.robot_config_status.setStyleSheet("padding:7px;color:#087a26;font-weight:700;")
        self._log(message)

    def _on_point_move_failed(self, reason: str) -> None:
        self.robot_config_status.setText(f"移动到点位失败并停止：{reason}")
        self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;font-weight:700;")
        self._log(f"真实机械臂移动到固定点位失败：{reason}")

    def _start_reference_anchor_capture(self, scene: str, color: str = "红") -> None:
        if self.session.state.value != "idle" or any(thread is not None for thread in (
            self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread,
            self.profile_thread, self.mvs_read_thread, self.readiness_thread, self.io_thread,
            self.point_capture_thread, self.point_move_thread, self.contact_capture_thread,
        )):
            QMessageBox.critical(self, "不能采集物理锚点", "比赛会话或其他维护流程正在运行。"); return
        labels = {"blocks": f"Block/{color}色基准抓取点", "trays": f"Tray/{color}色基准放置点"}
        if scene not in labels or color not in COLORS:
            return
        self.contact_capture_thread = QThread(self); self.contact_capture_worker = RobotReferenceAnchorCaptureWorker(scene, color)
        self.contact_capture_worker.moveToThread(self.contact_capture_thread); self.contact_capture_thread.started.connect(self.contact_capture_worker.run)
        self.contact_capture_worker.finished.connect(self._on_reference_anchor_capture_finished); self.contact_capture_worker.failed.connect(self._on_reference_anchor_capture_failed)
        self.contact_capture_worker.finished.connect(self.contact_capture_thread.quit); self.contact_capture_worker.failed.connect(self.contact_capture_thread.quit)
        self.contact_capture_thread.finished.connect(self._cleanup_reference_anchor_capture_worker)
        for button in self.reference_anchor_buttons.values(): button.setEnabled(False)
        self.robot_config_status.setText(f"正在只读采集{labels[scene]}；机械臂不会运动……")
        self.contact_capture_thread.start(); self._refresh_authorization_button()

    def _on_reference_anchor_capture_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        self._reload_robot_config()
        message = f"已从机器人{value.get('robot_name')}只读并保存{value.get('label')}：{value.get('tcp_pose')}。"
        self.robot_config_status.setText(message); self.robot_config_status.setStyleSheet("padding:7px;color:#087a26;font-weight:700;")
        self._log(message); self.run_preflight()

    def _on_reference_anchor_capture_failed(self, reason: str) -> None:
        self.robot_config_status.setText(f"物理锚点采集失败，未保存：{reason}")
        self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;font-weight:700;")
        self._log(f"真实物理锚点只读采集失败：{reason}")

    def _cleanup_reference_anchor_capture_worker(self) -> None:
        thread, worker = self.contact_capture_thread, self.contact_capture_worker
        self.contact_capture_thread = None; self.contact_capture_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        for button in self.reference_anchor_buttons.values(): button.setEnabled(True)
        self._refresh_authorization_button()

    def _cleanup_point_capture_worker(self) -> None:
        thread, worker = self.point_capture_thread, self.point_capture_worker
        self.point_capture_thread = None; self.point_capture_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.capture_point_button.setEnabled(True); self._refresh_authorization_button()

    def _cleanup_point_move_worker(self) -> None:
        thread, worker = self.point_move_thread, self.point_move_worker
        self.point_move_thread = None; self.point_move_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.capture_point_button.setEnabled(True)
        for button in self.move_point_buttons.values():
            button.setEnabled(True)
        self._refresh_authorization_button()

    def _on_joint_unit_changed(self, _index: int) -> None:
        target = str(self.joint_unit_combo.currentData())
        rows = [[self.points_table.item(row, column).text() for column in range(1, 7)] for row in range(len(POINT_KEYS))]
        if any(any(value.strip() for value in values) and not all(value.strip() for value in values) for values in rows):
            self.joint_unit_combo.blockSignals(True); self.joint_unit_combo.setCurrentIndex(0 if self._joint_unit_key == "rad" else 1); self.joint_unit_combo.blockSignals(False)
            self.robot_config_status.setText("某个关节点未填完整六轴，不能切换单位。")
            return
        for row, values in enumerate(rows):
            converted = convert_joint_display(values, from_units=self._joint_unit_key, to_units=target)
            for column, value in enumerate(converted, start=1): self.points_table.item(row, column).setText(value)
        self._joint_unit_key = target

    def _save_robot_config(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            self.robot_config_status.setText("比赛、九点标定或真实视觉验证运行中，禁止修改机器人配置。")
            self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        if not self.robot_config_confirm.isChecked():
            self.robot_config_status.setText("请先勾选真实 ARCS数据确认。")
            self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;"); return
        point_values = {key: [self.points_table.item(row, column).text().strip() for column in range(1, 7)] for row, key in enumerate(POINT_KEYS)}
        try:
            save_robot_editor_values(
                REAL_CONFIG_DIR / "robot.json", REAL_CONFIG_DIR / "motion.json",
                tcp_name="flange_zero_tcp", tcp_values=[0.0] * 6,
                tcp_units="m_rad", point_values=point_values,
                joint_units=self._joint_unit_key,
            )
        except (RobotConfigInputError, OSError) as exc:
            self.robot_config_status.setText(f"保存失败：{exc}"); self.robot_config_status.setStyleSheet("padding:7px;color:#a00000;")
            return
        if self.session.state.value != "idle":
            self.session.revoke("机器人现场配置已修改")
            self.session_status.setText("授权已撤销：机器人现场配置已修改")
        self.robot_config_confirm.setChecked(False)
        self.robot_config_status.setText("保存成功；真实机器人总放行保持未通过，必须重新执行赛前检查。")
        self.robot_config_status.setStyleSheet("padding:7px;color:#087a26;")
        if hasattr(self, "robot_readiness_status"):
            self.robot_readiness_status.setText("● 未检查：机器人配置已修改，请重新执行只读检查")
            self.robot_readiness_status.setStyleSheet("padding:8px;color:#666;background:#eee;font-weight:700;")
        self._log("用户保存四个固定关节点；TCP固定为零法兰值，未连接或控制硬件。")
        self.run_preflight()

    def _calibration_page(self) -> QWidget:
        page = QWidget(); outer = QVBoxLayout(page); scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content); outer.addWidget(scroll)
        warning = QLabel(
            "真实九点标定会直接以70%速度 MoveJoint到所选拍照点；九个采样点之间保持50%。不要求路线验收；程序不提供碰撞规划，也不写吸盘IO。\n"
            "点击开始后会自动调用 RuntimeMachine.start()；不会调用 runProgram()，不会自动运行已加载的ARCS程序。\n"
            "勾选一次安全确认并开始后，将连续完成全部9点的移动、稳定等待、拍照与采样，不再逐点要求确认。\n"
            "按操作者决定，九点拟合成功后会跳过方向验证并直接写入所选 blocks/trays 拍照点；首次抓取必须低速观察方向。\n"
            "开始前必须确认当前姿态到目标点及九点范围内无人员、障碍物，真实机器人/MVS/TCP身份全部一致。\n"
            "九点采用所选目标颜色的单色唯一性检测，不要求 blocks/trays 六色参数已批准；未找到唯一目标时立即停止。"
        )
        warning.setWordWrap(True); warning.setStyleSheet("padding:10px;background:#ffd6d6;font-weight:700;")
        layout.addWidget(warning)

        readiness = QGroupBox("机械臂控制条件指示（只读，不运动）")
        readiness_layout = QHBoxLayout(readiness)
        self.robot_readiness_button = QPushButton("只读检查机械臂是否可进入标定")
        self.robot_readiness_button.clicked.connect(self._start_robot_readiness_check)
        self.robot_readiness_status = QLabel("● 未检查：请先在 ARCS手动上电；RuntimeMachine可由九点流程自动启动")
        self.robot_readiness_status.setWordWrap(True)
        self.robot_readiness_status.setStyleSheet("padding:8px;color:#666;background:#eee;font-weight:700;")
        readiness_layout.addWidget(self.robot_readiness_button); readiness_layout.addWidget(self.robot_readiness_status, 1)
        layout.addWidget(readiness)

        settings = QGroupBox("九点参数（工具坐标系，单位 mm；速度比例固定50%，blend=0）")
        grid = QGridLayout(settings)
        for column, text in enumerate(("场景", "X步长", "Y步长", "目标颜色", "采集方式")):
            grid.addWidget(QLabel(text), 0, column)
        self.calibration_step_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        self.calibration_color_combos: dict[str, QComboBox] = {}
        self.calibration_auto_labels: dict[str, QLabel] = {}
        for row, scene in enumerate(SCENES, start=1):
            grid.addWidget(QLabel("方块 blocks" if scene == "blocks" else "托盘 trays"), row, 0)
            x_edit, y_edit = QLineEdit(), QLineEdit(); grid.addWidget(x_edit, row, 1); grid.addWidget(y_edit, row, 2)
            color = QComboBox()
            for name in COLORS: color.addItem(name, name)
            grid.addWidget(color, row, 3)
            auto_label = QLabel("一次连续完成"); grid.addWidget(auto_label, row, 4)
            self.calibration_step_edits[scene] = (x_edit, y_edit); self.calibration_color_combos[scene] = color; self.calibration_auto_labels[scene] = auto_label
        grid.addWidget(QLabel("直线加速度(m/s²)"), 3, 0); self.calibration_accel_edit = QLineEdit(); grid.addWidget(self.calibration_accel_edit, 3, 1)
        grid.addWidget(QLabel("直线速度(m/s)"), 3, 2); self.calibration_velocity_edit = QLineEdit(); grid.addWidget(self.calibration_velocity_edit, 3, 3)
        grid.addWidget(QLabel("到位稳定(s)"), 4, 0); self.calibration_settle_edit = QLineEdit(); grid.addWidget(self.calibration_settle_edit, 4, 1)
        save_settings = QPushButton("保存九点参数（不连接硬件）"); save_settings.clicked.connect(self._save_calibration_settings); grid.addWidget(save_settings, 4, 3, 1, 2)
        layout.addWidget(settings)

        controls_group = QGroupBox("生成最新真实标定文件")
        controls = QGridLayout(controls_group)
        self.calibration_scene_combo = QComboBox(); self.calibration_scene_combo.addItem("方块 blocks", "blocks"); self.calibration_scene_combo.addItem("托盘 trays", "trays")
        self.calibration_scene_combo.currentIndexChanged.connect(self._update_calibration_grid)
        for scene, edits in self.calibration_step_edits.items():
            for edit in edits:
                edit.textChanged.connect(lambda _text, value=scene: self._update_calibration_grid_for_scene(value))
        self.calibration_automatic_check = QCheckBox(); self.calibration_automatic_check.setChecked(True); self.calibration_automatic_check.hide()
        self.calibration_safety_confirm = QCheckBox("我已确认当前姿态到目标点及九点范围无人员和障碍物；吸盘未持物；本程序不提供碰撞规划")
        self.start_calibration_button = QPushButton("直接到所选拍照点并生成/重新生成九点文件")
        self.start_calibration_button.clicked.connect(self._start_calibration)
        self.accept_calibration_button = QPushButton(); self.accept_calibration_button.setEnabled(False); self.accept_calibration_button.hide()
        self.stop_calibration_button = QPushButton("停止并取消本套九点"); self.stop_calibration_button.setEnabled(False); self.stop_calibration_button.clicked.connect(self._stop_calibration)
        controls.addWidget(QLabel("场景"), 0, 0); controls.addWidget(self.calibration_scene_combo, 0, 1); controls.addWidget(QLabel("自动连续完成全部9点"), 0, 2)
        controls.addWidget(self.calibration_safety_confirm, 1, 0, 1, 3)
        controls.addWidget(self.start_calibration_button, 2, 0, 1, 2); controls.addWidget(self.stop_calibration_button, 2, 2)
        layout.addWidget(controls_group)

        tables = QHBoxLayout()
        self.calibration_grid_table = QTableWidget(9, 5)
        self.calibration_grid_table.setHorizontalHeaderLabels(("点", "相机X", "相机Y", "工具X预期", "工具Y预期")); self.calibration_grid_table.verticalHeader().setVisible(False)
        self.calibration_grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.calibration_grid_table.setMaximumHeight(250)
        self.calibration_points_table = QTableWidget(9, 7)
        self.calibration_points_table.setHorizontalHeaderLabels(("点", "状态", "像素U", "像素V", "工具X", "工具Y", "帧号")); self.calibration_points_table.verticalHeader().setVisible(False)
        self.calibration_points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.calibration_points_table.setMaximumHeight(250)
        tables.addWidget(self.calibration_grid_table); tables.addWidget(self.calibration_points_table); layout.addLayout(tables)

        result_group = QGroupBox("标定结果（九点完成后自动写入当前场景拍照点）")
        result_layout = QGridLayout(result_group)
        self.calibration_candidate_label = QLabel("尚未生成候选文件"); self.calibration_candidate_label.setWordWrap(True)
        result_layout.addWidget(self.calibration_candidate_label, 0, 0, 1, 4)
        validation_specs = (
            ("x_positive", "目标沿工具 +X移动10 mm后拍照"),
            ("y_positive", "目标沿工具 +Y移动10 mm后拍照"),
            ("angle_zero", "目标摆正0°后拍照"),
            ("angle_positive_10deg", "目标沿工具 +RZ旋转10°后拍照"),
        )
        self.calibration_validation_buttons: dict[str, QPushButton] = {}
        for column, (kind, text) in enumerate(validation_specs):
            button = QPushButton(text); button.setEnabled(False); button.hide()
            self.calibration_validation_buttons[kind] = button
        self.calibration_validation_label = QLabel("方向验证：按操作者决定跳过；九点拟合成功后自动批准并启用。")
        result_layout.addWidget(self.calibration_validation_label, 1, 0, 1, 3)
        open_folder = QPushButton("打开正式标定文件夹"); open_folder.clicked.connect(self._open_calibration_folder); result_layout.addWidget(open_folder, 1, 3)
        layout.addWidget(result_group)

        self.calibration_image_label = QLabel("真实采集后显示最新标注图路径")
        self.calibration_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.calibration_image_label.setMinimumHeight(80)
        self.calibration_image_label.setStyleSheet("background:#222;color:#ddd;padding:8px;"); layout.addWidget(self.calibration_image_label)
        self.calibration_status = QLabel("尚未加载九点参数"); self.calibration_status.setWordWrap(True); self.calibration_status.setStyleSheet("padding:7px;color:#555;")
        layout.addWidget(self.calibration_status); layout.addStretch()
        self._reload_calibration_settings(); self._update_calibration_grid()
        return page

    def _reload_calibration_settings(self) -> None:
        try:
            value = load_calibration_settings(REAL_CONFIG_DIR / "motion.json")
        except CalibrationConfigError as exc:
            self.calibration_status.setText(f"九点参数加载失败：{exc}"); return
        self.calibration_accel_edit.setText("" if value.get("linear_acceleration_m_s2") == "UNSET" else str(value.get("linear_acceleration_m_s2")))
        self.calibration_velocity_edit.setText("" if value.get("linear_velocity_m_s") == "UNSET" else str(value.get("linear_velocity_m_s")))
        self.calibration_settle_edit.setText(str(value.get("settle_s", 1.0)))
        for scene in SCENES:
            config = value[scene]; x_edit, y_edit = self.calibration_step_edits[scene]
            x_edit.setText(str(config["step_x_mm"])); y_edit.setText(str(config["step_y_mm"]))
            index = self.calibration_color_combos[scene].findData(config["target_color"]); self.calibration_color_combos[scene].setCurrentIndex(max(0, index))
            self.calibration_auto_labels[scene].setText("一次连续完成")
        self.calibration_candidate_label.setText(self._loaded_calibration_summary())
        self._refresh_calibration_automatic()

    @staticmethod
    def _loaded_calibration_summary() -> str:
        lines = ["当前实际载入的正式九点标定："]
        for scene, label in (("blocks", "Block"), ("trays", "Tray")):
            path = REAL_CALIBRATION_DIR / scene / f"9point_{scene}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                created = str(payload.get("created_at") or "").strip()
                if created:
                    generated_at = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    generated_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S（按文件修改时间）")
                lines.append(
                    f"{label}：生成于 {generated_at}，标定ID={payload.get('calibration_id', '未知')}，"
                    f"approved={payload.get('approved', False)}"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                lines.append(f"{label}：正式标定文件未能载入（{exc}）")
        return "\n".join(lines)

    def _persist_calibration_settings(self) -> set[str]:
        scenes = {
            scene: {"step_x_mm": self.calibration_step_edits[scene][0].text(), "step_y_mm": self.calibration_step_edits[scene][1].text(), "target_color": self.calibration_color_combos[scene].currentData()}
            for scene in SCENES
        }
        return save_calibration_settings(
            REAL_CONFIG_DIR / "motion.json", linear_acceleration_m_s2=self.calibration_accel_edit.text(),
            linear_velocity_m_s=self.calibration_velocity_edit.text(), settle_s=self.calibration_settle_edit.text(), scenes=scenes,
        )

    def _save_calibration_settings(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            self.calibration_status.setText("比赛、九点标定或真实视觉验证运行中，禁止修改九点参数。"); return
        try:
            changed = self._persist_calibration_settings()
        except (CalibrationConfigError, OSError) as exc:
            self.calibration_status.setText(f"九点参数保存失败：{exc}"); return
        self.calibration_status.setText(f"九点参数已保存；速度比例固定50%。失效场景：{', '.join(sorted(changed)) if changed else '无'}")
        self._reload_calibration_settings(); self._update_calibration_grid(); self.run_preflight()

    def _start_robot_readiness_check(self) -> None:
        if any(thread is not None for thread in (self.readiness_thread, self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread)):
            return
        self.readiness_thread = QThread(self); self.readiness_worker = RobotReadinessWorker()
        self.readiness_worker.moveToThread(self.readiness_thread); self.readiness_thread.started.connect(self.readiness_worker.run)
        self.readiness_worker.finished.connect(self._on_robot_readiness_ok); self.readiness_worker.failed.connect(self._on_robot_readiness_failed)
        self.readiness_worker.finished.connect(self.readiness_thread.quit); self.readiness_worker.failed.connect(self.readiness_thread.quit)
        self.readiness_thread.finished.connect(self._cleanup_robot_readiness)
        self.robot_readiness_button.setEnabled(False)
        self.robot_readiness_status.setText("● 正在只读连接并核对身份、Running、安全状态、执行队列和活动TCP……")
        self.robot_readiness_status.setStyleSheet("padding:8px;color:#7a5200;background:#fff2bf;font-weight:700;")
        self.readiness_thread.start(); self._refresh_authorization_button()

    def _on_robot_readiness_ok(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        self.robot_readiness_status.setText(
            f"● 绿色：机械臂控制条件通过（本次仅只读、未运动）｜机器人={value.get('robot_name')}｜"
            f"RobotMode={value.get('robot_mode')}｜SafetyMode={value.get('safety_mode')}｜exec_id={value.get('exec_id')}｜"
            f"Runtime={value.get('runtime_state')}｜活动TCP={value.get('active_tcp')}"
        )
        self.robot_readiness_status.setStyleSheet("padding:8px;color:#087a26;background:#d7f5dc;font-weight:700;")

    def _on_robot_readiness_failed(self, reason: str) -> None:
        self.robot_readiness_status.setText(f"● 红色：当前不能进入标定｜{reason}")
        self.robot_readiness_status.setStyleSheet("padding:8px;color:#a00000;background:#ffd3d3;font-weight:700;")

    def _cleanup_robot_readiness(self) -> None:
        thread, worker = self.readiness_thread, self.readiness_worker
        self.readiness_thread = None; self.readiness_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.robot_readiness_button.setEnabled(True); self._refresh_authorization_button()

    def _refresh_calibration_automatic(self) -> None:
        self.calibration_automatic_check.setChecked(True)

    def _update_calibration_grid(self, _index: int = 0) -> None:
        scene = str(self.calibration_scene_combo.currentData())
        try:
            grid = build_grid(self.calibration_step_edits[scene][0].text(), self.calibration_step_edits[scene][1].text())
        except Exception:
            for row in range(9):
                for column in range(5):
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.calibration_grid_table.setItem(row, column, item)
            self._refresh_calibration_automatic()
            return
        for row, point in enumerate(grid):
            for column, value in enumerate((point.index, point.camera_x_mm, point.camera_y_mm, point.expected_tool_x_mm, point.expected_tool_y_mm)):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.calibration_grid_table.setItem(row, column, item)
        self._refresh_calibration_automatic()

    def _update_calibration_grid_for_scene(self, scene: str) -> None:
        if str(self.calibration_scene_combo.currentData()) == scene:
            self._update_calibration_grid()

    def _start_calibration(self) -> None:
        if any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            QMessageBox.critical(self, "不能开始九点", "比赛、九点标定或真实视觉验证正在运行。"); return
        if not self.calibration_safety_confirm.isChecked():
            QMessageBox.critical(self, "不能开始九点", "请先确认全部直接运动路径无人员和障碍物。"); return
        try:
            self._persist_calibration_settings()
        except (CalibrationConfigError, OSError) as exc:
            QMessageBox.critical(self, "九点参数无效", str(exc)); return
        if self.session.state.value != "idle":
            self.session.revoke("进入真实九点维护模式")
            self.session_status.setText("授权已撤销：进入真实九点维护模式")
        scene = str(self.calibration_scene_combo.currentData())
        self.calibration_candidate = None; self.calibration_validation = {}
        self.calibration_validation_label.setText("方向验证：按操作者决定跳过；正在生成九点。")
        for button in self.calibration_validation_buttons.values(): button.setEnabled(False)
        for row in range(9):
            for column in range(7): self.calibration_points_table.setItem(row, column, QTableWidgetItem(""))
        self.calibration_thread = QThread(self); self.calibration_worker = CalibrationWorker(scene=scene, automatic=True)
        self.calibration_worker.moveToThread(self.calibration_thread)
        self.calibration_thread.started.connect(self.calibration_worker.run)
        self.calibration_worker.progress.connect(self._on_calibration_progress)
        self.calibration_worker.candidate_ready.connect(self._on_calibration_candidate)
        self.calibration_worker.finished.connect(self.calibration_thread.quit); self.calibration_worker.failed.connect(self._on_calibration_failed); self.calibration_worker.failed.connect(self.calibration_thread.quit)
        self.calibration_thread.finished.connect(self._cleanup_calibration_worker)
        self.start_calibration_button.setEnabled(False); self.stop_calibration_button.setEnabled(True); self.calibration_safety_confirm.setChecked(False)
        self.calibration_status.setText("正在连接并只读核对真实机器人和MVS身份……")
        self.calibration_thread.start()

    def _accept_calibration_step(self) -> None:
        if self.calibration_worker is not None:
            self.accept_calibration_button.setEnabled(False); self.calibration_worker.accept_current_step()

    def _stop_calibration(self) -> None:
        if self.calibration_worker is not None:
            self.calibration_worker.request_stop(); self.stop_calibration_button.setEnabled(False); self.calibration_status.setText("已请求停止；本套九点作废，不自动续跑。")

    def _on_calibration_confirmation(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {"message": str(payload)}
        self.calibration_status.setText(str(value.get("message", "等待人工确认"))); self.accept_calibration_button.setEnabled(True)

    def _on_calibration_progress(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {"message": str(payload)}
        if isinstance(value.get("annotated_image_path"), str):
            self._show_visual_result(value, self.calibration_status, self.calibration_image_label)
            return
        self.calibration_status.setText(str(value.get("message", value)))
        sample = value.get("sample")
        if isinstance(sample, dict) and isinstance(sample.get("index"), int):
            row = int(sample["index"]) - 1
            values = (sample["index"], "已采集", sample.get("pixel_u"), sample.get("pixel_v"), sample.get("tool_x_mm"), sample.get("tool_y_mm"), sample.get("frame_number"))
            for column, item in enumerate(values): self.calibration_points_table.setItem(row, column, QTableWidgetItem(str(item)))
            self._show_calibration_image(str(sample.get("image_path", "")))
        elif value.get("phase") == "precheck":
            self._show_calibration_image(str(value.get("image_path", "")))

    def _on_calibration_candidate(self, payload: object) -> None:
        if not isinstance(payload, dict): return
        self.calibration_candidate = dict(payload); self.calibration_validation = {}
        self.calibration_candidate_label.setText(
            f"九点结果：{payload.get('calibration_id')}\n{payload.get('candidate_path')}\nRMS={payload.get('rms_error_mm')} mm，最大误差={payload.get('max_error_mm')} mm；正在自动写入正式标定"
        )
        self._approve_calibration_candidate()

    def _on_calibration_failed(self, reason: str) -> None:
        self.calibration_status.setText(f"九点失败并停止：{reason}"); self._log(f"真实九点失败：{reason}")

    def _cleanup_calibration_worker(self) -> None:
        thread, worker = self.calibration_thread, self.calibration_worker
        self.calibration_thread = None; self.calibration_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.start_calibration_button.setEnabled(True); self.accept_calibration_button.setEnabled(False); self.stop_calibration_button.setEnabled(False)
        self._refresh_authorization_button()

    def _start_calibration_validation(self, kind: str) -> None:
        if self.calibration_candidate is None or any(thread is not None for thread in (self.validation_thread, self.calibration_thread, self.competition_thread, self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread)):
            return
        session_id = str(self.calibration_candidate["session_id"]); scene = str(self.calibration_candidate["scene"])
        self.validation_thread = QThread(self); self.validation_worker = CalibrationValidationWorker(session_id=session_id, scene=scene, validation_kind=kind)
        self.validation_worker.moveToThread(self.validation_thread); self.validation_thread.started.connect(self.validation_worker.run)
        self.validation_worker.visual.connect(self._on_calibration_visual_result)
        self.validation_worker.finished.connect(self._on_calibration_validation_result); self.validation_worker.failed.connect(self._on_calibration_validation_failed)
        self.validation_worker.finished.connect(self.validation_thread.quit); self.validation_worker.failed.connect(self.validation_thread.quit)
        self.validation_thread.finished.connect(self._cleanup_validation_worker)
        for button in self.calibration_validation_buttons.values(): button.setEnabled(False)
        self.calibration_status.setText(f"正在触发真实新帧验证：{kind}"); self.validation_thread.start()

    def _on_calibration_validation_result(self, payload: object) -> None:
        if not isinstance(payload, dict): return
        kind = str(payload.get("validation_kind")); detection = payload.get("detection")
        if not isinstance(detection, dict): return
        if kind in {"x_positive", "y_positive"}:
            self.calibration_validation[kind] = {"dx_mm": float(detection["dx_tool_m"]) * 1000.0, "dy_mm": float(detection["dy_tool_m"]) * 1000.0}
        else:
            self.calibration_validation[kind] = math.degrees(float(detection["r_image_rad"]))
        self.calibration_validation_label.setText(f"方向验证：{len(self.calibration_validation)}/4　{json.dumps(self.calibration_validation, ensure_ascii=False)}")
        self._show_calibration_image(str(payload.get("image_path", "")))

    def _on_calibration_validation_failed(self, reason: str) -> None:
        self.calibration_status.setText(f"方向验证失败：{reason}")

    def _on_calibration_visual_result(self, payload: object) -> None:
        self._show_visual_result(payload, self.calibration_status, self.calibration_image_label)

    def _cleanup_validation_worker(self) -> None:
        thread, worker = self.validation_thread, self.validation_worker
        self.validation_thread = None; self.validation_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        if self.calibration_candidate is not None:
            for button in self.calibration_validation_buttons.values(): button.setEnabled(True)
        self._refresh_authorization_button()

    def _approve_calibration_candidate(self) -> None:
        if self.calibration_candidate is None:
            QMessageBox.critical(self, "不能批准", "尚未生成九点候选文件。"); return
        scene = str(self.calibration_candidate["scene"])
        try:
            approved = approve_candidate_without_direction_validation(
                Path(str(self.calibration_candidate["candidate_path"])),
                REAL_CALIBRATION_DIR / scene / f"9point_{scene}.json",
                EVIDENCE_DIR / "calibration_archive" / scene,
            )
            mark_automatic_verified(REAL_CONFIG_DIR / "motion.json", scene, True)
        except Exception as exc:
            QMessageBox.critical(self, "批准失败", str(exc)); return
        if self.session.state.value != "idle": self.session.revoke("真实九点标定已更新")
        self.calibration_validation_label.setText("方向验证：已按操作者决定跳过。")
        self.calibration_candidate_label.setText(
            f"已自动批准并启用：{approved['calibration_id']}\n"
            f"{REAL_CALIBRATION_DIR / scene / f'9point_{scene}.json'}\n"
            f"RMS={approved['rms_error_mm']} mm，最大误差={approved['max_error_mm']} mm；未做方向验证"
        )
        self.calibration_status.setText(
            f"已自动批准并写入 {scene}拍照点标定：{approved['calibration_id']}（方向验证已跳过）；"
            "已保留该场景先前保存的六色物理基准。"
            "仅当六色目标在整个九点过程中完全未移动时，这些基准才仍然有效。"
        )
        self._reload_calibration_settings(); self._reload_robot_config(); self.run_preflight()

    def _open_calibration_folder(self) -> None:
        scene = str(self.calibration_scene_combo.currentData()); path = REAL_CALIBRATION_DIR / scene; path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _show_calibration_image(self, path_text: str) -> None:
        path = resolve_project_path(path_text)
        if not path.is_file(): return
        pixmap = QPixmap(str(path)); self.calibration_image_label.setPixmap(pixmap.scaled(620, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @staticmethod
    def _show_visual_result(payload: object, status_label: QLabel, image_label: QLabel) -> None:
        value = payload if isinstance(payload, dict) else {}
        summary = value.get("detection_summary") if isinstance(value.get("detection_summary"), dict) else {}
        reports = summary.get("colors") if isinstance(summary.get("colors"), list) else []
        parts: list[str] = []
        for report in reports:
            if not isinstance(report, dict):
                continue
            color, status = report.get("color", "?"), report.get("status", "unknown")
            selected = report.get("selected") if isinstance(report.get("selected"), dict) else None
            if status == "success" and selected is not None:
                center = selected.get("center", ["?", "?"])
                parts.append(f"{color}成功：中心({center[0]}, {center[1]})，面积{selected.get('area_px')}，置信度{selected.get('confidence')}")
            else:
                candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
                parts.append(f"{color}失败：{status}，画面候选{len(candidates)}个")
        success = value.get("success") is True and summary.get("success") is True
        message = str(value.get("message", "识别成功" if success else "识别失败"))
        status_label.setText(message + ("\n" + "；".join(parts) if parts else ""))
        status_label.setStyleSheet(
            "padding:8px;font-weight:700;color:#087a26;background:#d7f5dc;" if success
            else "padding:8px;font-weight:700;color:#a00000;background:#ffd3d3;"
        )
        path = resolve_project_path(str(value.get("annotated_image_path", "")))
        if path.is_file():
            pixmap = QPixmap(str(path))
            image_label.setPixmap(pixmap.scaled(760, 360, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            image_label.setToolTip(str(path))

    def _logs_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(200)
        self.log_view.setStyleSheet("background:#111;color:#ddd;font-family:Consolas,monospace;")
        layout.addWidget(self.log_view); return page

    def _voice_interaction_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        explanation = QLabel(
            "本页只连接 config/real/endpoints.json 中配置的 AI语音盒子，不连接机械臂和相机，也不会触发比赛动作。\n"
            "“开始识别”后请对语音盒子的麦克风说话；盒子返回的原始识别文字会显示为[语音识别]。"
            "比赛页发送的文字指令也会同步显示为[文字控制]。"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("padding:12px;background:#dcecff;font-weight:700;")
        layout.addWidget(explanation)

        controls = QGroupBox("语音输入（说话 → 文字）")
        controls_layout = QGridLayout(controls)
        self.voice_health_button = QPushButton("① 检测语音盒子连接")
        self.voice_health_button.clicked.connect(lambda: self._start_voice_interaction("health"))
        self.voice_wakeup_check = QCheckBox("本次先说“小具同学”唤醒，再说指令")
        self.voice_listen_button = QPushButton("② 开始识别并显示文字")
        self.voice_listen_button.setMinimumHeight(48)
        self.voice_listen_button.setStyleSheet("font-size:16px;font-weight:700;background:#1769aa;color:white;padding:8px;")
        self.voice_listen_button.clicked.connect(lambda: self._start_voice_interaction("listen"))
        controls_layout.addWidget(self.voice_health_button, 0, 0)
        controls_layout.addWidget(self.voice_wakeup_check, 0, 1)
        controls_layout.addWidget(self.voice_listen_button, 1, 0, 1, 2)
        layout.addWidget(controls)

        self.voice_status = QLabel("尚未测试。请先检测连接，或直接点击“开始识别并显示文字”。")
        self.voice_status.setWordWrap(True)
        self.voice_status.setStyleSheet("padding:10px;background:#f1f1f1;font-weight:700;")
        layout.addWidget(self.voice_status)
        layout.addWidget(QLabel("最近收到的控制文字（语音识别 / 文字控制）："))
        self.voice_recognized_text = QPlainTextEdit()
        self.voice_recognized_text.setReadOnly(True)
        self.voice_recognized_text.setPlaceholderText("语音识别或文字控制发送成功后，内容会显示在这里……")
        self.voice_recognized_text.setMinimumHeight(140)
        self.voice_recognized_text.setStyleSheet("font-size:22px;font-weight:700;background:white;padding:12px;")
        layout.addWidget(self.voice_recognized_text)

        tts = QGroupBox("语音输出（文字 → 盒子播报）")
        tts_layout = QHBoxLayout(tts)
        self.voice_tts_edit = QLineEdit("语音交互测试成功")
        self.voice_tts_edit.setPlaceholderText("输入希望语音盒子播报的文字")
        self.voice_speak_button = QPushButton("让语音盒子播报")
        self.voice_speak_button.clicked.connect(lambda: self._start_voice_interaction("speak"))
        self.voice_tts_edit.returnPressed.connect(lambda: self._start_voice_interaction("speak"))
        tts_layout.addWidget(self.voice_tts_edit); tts_layout.addWidget(self.voice_speak_button)
        layout.addWidget(tts)
        layout.addStretch()
        return page

    def _start_voice_interaction(self, action: str) -> None:
        if self.voice_thread is not None:
            self.voice_status.setText("语音盒子正在执行上一项操作，请等待完成。")
            return
        if self.competition_thread is not None:
            self.voice_status.setText("比赛或单组抓放正在运行，不能同时占用语音盒子。")
            return
        if action == "speak" and not self.voice_tts_edit.text().strip():
            self.voice_status.setText("请输入希望语音盒子播报的文字。")
            return
        self.voice_thread = QThread(self)
        self.voice_worker = VoiceInteractionWorker(
            action,  # type: ignore[arg-type]
            wakeup_required=self.voice_wakeup_check.isChecked(),
            timeout_s=30.0,
            text=self.voice_tts_edit.text(),
        )
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.finished.connect(self._on_voice_interaction_finished)
        self.voice_worker.failed.connect(self._on_voice_interaction_failed)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_worker.failed.connect(self.voice_thread.quit)
        self.voice_thread.finished.connect(self._cleanup_voice_interaction)
        self._set_voice_controls_enabled(False)
        if action == "health":
            message = "正在连接并核对语音盒子身份与 ASR/TTS 能力……"
        elif action == "listen":
            self.voice_recognized_text.clear()
            message = (
                "正在等待唤醒词和语音指令，请对语音盒子说话……"
                if self.voice_wakeup_check.isChecked()
                else "正在录音识别，请现在对语音盒子说话……"
            )
        else:
            message = "正在请求语音盒子播报，请听盒子是否发声……"
        self.voice_status.setText(message)
        self.voice_status.setStyleSheet("padding:10px;color:#7a5200;background:#fff2bf;font-weight:700;")
        self.voice_thread.start()
        self._log(f"启动语音盒子独立测试：{action}")
        self._refresh_authorization_button(); self._refresh_direct_assembly_controls()

    def _on_voice_interaction_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        action = value.get("action")
        if action == "listen":
            text = str(value.get("recognized_text", ""))
            self.voice_recognized_text.setPlainText(f"[语音识别]\n{text}")
            message = f"识别成功：语音盒子听到“{text}”"
        elif action == "speak":
            message = f"播报请求成功，盒子回报已播出：“{value.get('spoken_text', '')}”。请同时以现场实际听到为准。"
        else:
            health = value.get("health") if isinstance(value.get("health"), dict) else {}
            service = health.get("service", "arm_speech_service")
            version = health.get("version", "未知")
            message = f"连接成功：服务={service}，版本={version}，ASR/TTS 身份检查通过。"
        self.voice_status.setText(message)
        self.voice_status.setStyleSheet("padding:10px;color:#087a26;background:#d7f5dc;font-weight:700;")
        self._log(message)

    def _on_voice_interaction_failed(self, reason: str) -> None:
        message = f"语音盒子测试失败：{reason}"
        self.voice_status.setText(message)
        self.voice_status.setStyleSheet("padding:10px;color:#a00000;background:#ffd3d3;font-weight:700;")
        self._log(message)

    def _cleanup_voice_interaction(self) -> None:
        thread, worker = self.voice_thread, self.voice_worker
        self.voice_thread = None; self.voice_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self._set_voice_controls_enabled(True)
        self._refresh_authorization_button(); self._refresh_direct_assembly_controls()

    def _set_voice_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.voice_health_button,
            self.voice_wakeup_check,
            self.voice_listen_button,
            self.voice_tts_edit,
            self.voice_speak_button,
        ):
            widget.setEnabled(enabled)

    def _start_manual_io(self, operation: str) -> None:
        if self.session.state.value != "idle" or any(thread is not None for thread in (
            self.competition_thread, self.calibration_thread, self.validation_thread,
            self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread,
        )):
            QMessageBox.critical(self, "不能执行 IO", "比赛会话或其他维护流程正在运行；请先停止并撤销授权。")
            return
        descriptions = {
            "toggle_aperture": "连接配置中的唯一 AUBO，读取 DO0 后写入相反电平并回读；这会真实开/关光圈。",
            "suction_on": "连接配置中的唯一 AUBO，确保TOOL_IO[1]为输出并长期写1，再把TOOL_IO[0]写0并回读；这会真实启动吸盘。",
            "suction_off": "连接配置中的唯一 AUBO，保持TOOL_IO[1]=1，把TOOL_IO[0]写1并回读；这会停止吸取但不关闭长期使能。",
        }
        description = descriptions.get(operation)
        if description is None:
            QMessageBox.critical(self, "不能执行 IO", "未知 IO 操作。")
            return
        answer = QMessageBox.question(
            self,
            "确认真实 IO 操作",
            f"{description}\n\n程序不会上电、startup或运动。请确认现场人员已远离机构，并已在 ARCS 手动置于允许状态。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.io_thread = QThread(self); self.io_worker = ManualIoWorker(operation)  # type: ignore[arg-type]
        self.io_worker.moveToThread(self.io_thread)
        self.io_thread.started.connect(self.io_worker.run)
        self.io_worker.finished.connect(self._on_manual_io_finished)
        self.io_worker.failed.connect(self._on_manual_io_failed)
        self.io_worker.finished.connect(self.io_thread.quit)
        self.io_worker.failed.connect(self.io_thread.quit)
        self.io_thread.finished.connect(self._cleanup_manual_io_worker)
        self.io_status.setText("正在连接唯一机器人并核对身份/安全状态/执行队列；尚未确认写入成功……")
        self._set_manual_io_buttons_enabled(False)
        self.io_thread.start()
        self._log(f"用户现场确认并启动真实 IO 单步操作：{operation}")
        self._refresh_authorization_button()

    def _on_manual_io_finished(self, result: object) -> None:
        value = result if isinstance(result, dict) else {}
        state = "打开" if value.get("enabled") else "关闭"
        if value.get("device") == "吸盘":
            message = (
                f"真实回读成功：TOOL_IO[{value.get('enable_index')}]=1保持使能，"
                f"TOOL_IO[{value.get('index')}]已{state}吸取；机器人={value.get('robot_name')}。"
            )
        else:
            message = f"真实回读成功：{value.get('device')} DO{value.get('index')} 已{state}；机器人={value.get('robot_name')}。请现场核对物理动作。"
        self.io_status.setText(message); self.io_status.setStyleSheet("padding:10px;color:#087a26;font-weight:700;")
        self._log(message)

    def _on_manual_io_failed(self, reason: str) -> None:
        self.io_status.setText(f"IO 操作失败，状态不可假定：{reason}")
        self.io_status.setStyleSheet("padding:10px;color:#a00000;font-weight:700;")
        self._log(f"真实 IO 单步操作失败：{reason}")

    def _cleanup_manual_io_worker(self) -> None:
        thread, worker = self.io_thread, self.io_worker
        self.io_thread = None; self.io_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self._set_manual_io_buttons_enabled(True)
        self._refresh_authorization_button()

    def _set_manual_io_buttons_enabled(self, enabled: bool) -> None:
        for button in (self.aperture_toggle_button, self.suction_on_button, self.suction_off_button):
            button.setEnabled(enabled)

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(f"{datetime.now().isoformat(timespec='milliseconds')}  {message}")

    def run_preflight(self) -> None:
        self.checks = run_static_preflight()
        self.preflight_table.setRowCount(len(self.checks))
        colors = {"PASS": QColor("#d7f5dc"), "WARN": QColor("#fff2bf"), "FAIL": QColor("#ffd3d3")}
        category_names = {"配置": "基础配置", "端口": "设备连接", "软件环境": "程序环境", "云服务": "大模型", "真实标定": "九点标定", "完整性": "版本保护"}
        status_names = {"PASS": "通过", "WARN": "提醒", "FAIL": "未通过"}
        for row, check in enumerate(self.checks):
            title, _explanation, location = _friendly_preflight(check)
            values = (category_names.get(check.category, check.category), title, check.actual, check.expected, status_names[check.status], location, "是" if check.critical else "否")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setToolTip(value); item.setBackground(colors[check.status]); self.preflight_table.setItem(row, column, item)
        ready = competition_ready(self.checks)
        failed = sum(item.status == "FAIL" for item in self.checks)
        warned = sum(item.status == "WARN" for item in self.checks)
        self.preflight_status.setText("全部通过，可以申请比赛授权" if ready else f"禁止比赛：{failed} 个关键失败，{warned} 个警告")
        self.preflight_status.setStyleSheet(f"font-size:16px;font-weight:700;color:{'#087a26' if ready else '#a00000'};")
        if not ready and self.session.state.value != "idle":
            self.session.revoke("赛前检查不再满足")
            self.session_status.setText("授权已撤销：赛前检查不满足")
        first_problem = next((index for index, item in enumerate(self.checks) if item.status == "FAIL"), 0 if self.checks else -1)
        if first_problem >= 0:
            self.preflight_table.setCurrentCell(first_problem, 0)
            self._update_preflight_detail()
        self._refresh_authorization_button()
        self._log(f"静态赛前检查完成：ready={ready}, failures={failed}, warnings={warned}")

    def _refresh_authorization_button(self) -> None:
        maintenance_busy = self.calibration_thread is not None or self.validation_thread is not None or self.detector_thread is not None or self.profile_thread is not None or self.mvs_read_thread is not None or self.readiness_thread is not None or self.io_thread is not None or self.point_capture_thread is not None or self.point_move_thread is not None or self.contact_capture_thread is not None or self.voice_thread is not None
        self.authorize_button.setEnabled(competition_ready(self.checks) and self.authorization_check.isChecked() and self.direct_paths_check.isChecked() and not maintenance_busy)
        running = self.competition_thread is not None and self.competition_thread.isRunning()
        self.start_competition_button.setEnabled(self.session.state.value == "authorized" and not running and not maintenance_busy)
        self.stop_competition_button.setEnabled(running and isinstance(self.competition_worker, CompetitionWorker))

    def _authorize(self) -> None:
        try:
            self.session.authorize(competition_ready(self.checks))
        except SessionError as exc:
            QMessageBox.critical(self, "不能授权", str(exc)); return
        self.authorization_check.setChecked(False)
        self.direct_paths_check.setChecked(False)
        self.session_status.setText("本场比赛已授权；硬件状态变化会立即撤销")
        self._log("用户建立本场比赛自动执行授权。")
        self._refresh_authorization_button()

    def _refresh_direct_assembly_controls(self) -> None:
        running = self.competition_thread is not None or self.voice_thread is not None
        ready = hasattr(self, "direct_safety_check") and self.direct_safety_check.isChecked()
        if hasattr(self, "start_direct_assembly_button"):
            self.start_direct_assembly_button.setEnabled(bool(ready and not running))
            self.stop_direct_assembly_button.setEnabled(bool(running and isinstance(self.competition_worker, DirectAssemblyWorker)))

    def _start_direct_assembly(self) -> None:
        maintenance_busy = any(thread is not None for thread in (
            self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread,
            self.mvs_read_thread, self.readiness_thread, self.io_thread, self.point_capture_thread,
            self.point_move_thread, self.contact_capture_thread, self.voice_thread,
        ))
        if self.competition_thread is not None or maintenance_busy:
            QMessageBox.critical(self, "不能开始单组抓放", "比赛或其他真实硬件维护流程正在运行。")
            return
        if self.session.state.value != "idle":
            QMessageBox.critical(self, "不能开始单组抓放", "当前已有比赛会话授权；请重新启动总控建立独立调试会话。")
            return
        if not self.direct_safety_check.isChecked():
            QMessageBox.critical(self, "不能开始单组抓放", "必须先确认现场无人、无障碍物且六色托盘均在视野内。")
            return
        block_color = str(self.direct_block_color_combo.currentData())
        tray_color = str(self.direct_tray_color_combo.currentData())
        answer = QMessageBox.question(
            self,
            "确认真实单组抓放",
            f"将真实抓取“{block_color}”色方块并放入“{tray_color}”色托盘。\n\n"
            "机器人会自动启动 RuntimeMachine并执行 MoveJoint、MoveLine和吸盘IO；程序不提供碰撞规划。"
            "请确认急停可用、人员已撤离、红色抓取/放置基准TCP（含接触Z）正确。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.competition_thread = QThread(self)
        self.competition_worker = DirectAssemblyWorker(block_color=block_color, tray_color=tray_color)
        self.competition_worker.moveToThread(self.competition_thread)
        self.competition_thread.started.connect(self.competition_worker.run)
        self.competition_worker.progress.connect(self._on_direct_assembly_progress)
        self.competition_worker.finished.connect(self._on_direct_assembly_finished)
        self.competition_worker.failed.connect(self._on_direct_assembly_failed)
        self.competition_worker.finished.connect(self.competition_thread.quit)
        self.competition_worker.failed.connect(self.competition_thread.quit)
        self.competition_thread.finished.connect(self._cleanup_direct_assembly_worker)
        self.direct_assembly_status.setText(f"正在执行：{block_color}方块 → {tray_color}托盘")
        self.direct_assembly_log.clear()
        self.competition_thread.start()
        self._log(f"用户确认并启动真实单组抓放：{block_color}方块→{tray_color}托盘。")
        self._refresh_direct_assembly_controls(); self._refresh_text_controls(); self._refresh_authorization_button()

    def _stop_direct_assembly(self) -> None:
        worker = self.competition_worker
        if not isinstance(worker, DirectAssemblyWorker):
            return
        worker.request_stop()
        self.direct_assembly_status.setText("人工停止已请求；正在等待当前硬件命令安全退出。")
        self._log("用户请求人工停止单组抓放。")

    def _on_direct_assembly_progress(self, event: object) -> None:
        value = event if isinstance(event, dict) else {"phase": "event", "message": str(event)}
        phase = str(value.get("phase", "event")); message = str(value.get("message", ""))
        line = f"[{phase}] {message}"
        self.direct_assembly_status.setText(line)
        self.direct_assembly_log.appendPlainText(f"{datetime.now().isoformat(timespec='milliseconds')}  {line}")
        self._log(f"[direct_assembly] {line}")
        if phase == "visual_result":
            self._show_visual_result(value, self.direct_visual_status, self.direct_visual_image)

    def _on_direct_assembly_finished(self, payload: object) -> None:
        value = payload if isinstance(payload, dict) else {}
        message = f"单组抓放完成：{value.get('block_color')}方块 → {value.get('tray_color')}托盘；已返回比赛待机点。"
        self.direct_assembly_status.setText(message); self.direct_assembly_log.appendPlainText(message); self._log(message)

    def _on_direct_assembly_failed(self, reason: str) -> None:
        message = f"单组抓放失败并停止：{reason}"
        self.direct_assembly_status.setText(message); self.direct_assembly_log.appendPlainText(message); self._log(message)

    def _cleanup_direct_assembly_worker(self) -> None:
        thread, worker = self.competition_thread, self.competition_worker
        self.competition_thread = None; self.competition_worker = None
        if worker is not None: worker.deleteLater()
        if thread is not None: thread.deleteLater()
        self.direct_safety_check.setChecked(False)
        self._refresh_direct_assembly_controls(); self._refresh_text_controls(); self._refresh_authorization_button(); self.run_preflight()

    def _start_competition(self) -> None:
        if self.session.state.value != "authorized" or any(thread is not None for thread in (self.competition_thread, self.calibration_thread, self.validation_thread, self.detector_thread, self.profile_thread, self.readiness_thread, self.io_thread, self.point_capture_thread, self.point_move_thread, self.contact_capture_thread, self.voice_thread)):
            QMessageBox.critical(self, "不能启动", "当前没有有效比赛授权或流程已经运行。")
            return
        self.competition_thread = QThread(self)
        input_mode = (
            "countdown" if self.countdown_mode_button.isChecked()
            else "text" if self.text_mode_button.isChecked()
            else "voice"
        )
        self.competition_worker = CompetitionWorker(self.session, input_mode=input_mode)
        self.competition_worker.moveToThread(self.competition_thread)
        self.competition_thread.started.connect(self.competition_worker.run)
        self.competition_worker.progress.connect(self._on_competition_progress)
        self.competition_worker.finished.connect(self._on_competition_finished)
        self.competition_worker.failed.connect(self._on_competition_failed)
        self.competition_worker.finished.connect(self.competition_thread.quit)
        self.competition_worker.failed.connect(self.competition_thread.quit)
        self.competition_thread.finished.connect(self._cleanup_competition_worker)
        self.monitor_status.setText("正在执行真实服务身份核验……")
        self.tabs.setCurrentIndex(2)
        self.competition_thread.start()
        self._log(f"用户启动正式双任务卡流程；输入模式={input_mode}；Robot Worker开始只读身份核验。")
        self._refresh_text_controls()
        self._refresh_authorization_button()

    def _toggle_text_mode(self, checked: bool) -> None:
        if checked and self.countdown_mode_button.isChecked():
            self.countdown_mode_button.setChecked(False)
        self.text_mode_button.setText("切换到语音控制" if checked else "切换到文字控制")
        self.text_control_status.setText(
            "当前：文字控制。现在可以预先输入；建立授权并启动流程后，先发送“小具同学”，收到回复后再发送任务指令。"
            if checked else "当前：语音控制。比赛启动前可切换为文字控制。"
        )
        self._refresh_text_controls()

    def _toggle_countdown_mode(self, checked: bool) -> None:
        if checked and self.text_mode_button.isChecked():
            self.text_mode_button.setChecked(False)
        self.countdown_mode_button.setText("切换到语音控制" if checked else "切换到5秒倒计时控制")
        self.text_control_status.setText(
            "当前：自动倒计时控制。到达任务卡拍照点后，首次5秒自动唤醒、再5秒自动识别；第二张任务卡等待12秒自动识别。"
            if checked else "当前：语音控制。比赛启动前可切换为文字或倒计时控制。"
        )
        self._refresh_text_controls()

    def _refresh_text_controls(self) -> None:
        running = isinstance(self.competition_worker, CompetitionWorker)
        text_mode = self.text_mode_button.isChecked()
        self.text_mode_button.setEnabled(self.competition_worker is None)
        self.countdown_mode_button.setEnabled(self.competition_worker is None)
        self.text_input_edit.setEnabled(text_mode)
        self.send_text_button.setEnabled(running and text_mode)
        self.send_text_button.setToolTip("建立一次性比赛授权并启动正式流程后才可发送" if text_mode and not running else "")

    def _send_text_instruction(self) -> None:
        worker = self.competition_worker
        text = self.text_input_edit.text().strip()
        if not isinstance(worker, CompetitionWorker) or not self.text_mode_button.isChecked():
            self.text_control_status.setText("请先切换到文字控制并启动比赛流程。")
            return
        if not text:
            self.text_control_status.setText("文字指令不能为空。")
            return
        try:
            submitted = worker.submit_text(text)
        except (RuntimeError, ValueError) as exc:
            self.text_control_status.setText(str(exc))
            return
        self.text_input_edit.clear()
        self.text_control_status.setText(f"已发送：{submitted}")
        self.voice_recognized_text.setPlainText(f"[文字控制]\n{submitted}")
        self._log(f"[manual_text_submitted] {submitted}")

    def _stop_competition(self) -> None:
        worker = self.competition_worker
        if worker is None:
            return
        worker.request_stop()
        self.session_status.setText("人工停止已请求；比赛授权已撤销")
        self.monitor_status.setText("正在安全停止；不会自动续跑")
        self._log("用户请求人工停止；等待 Robot Worker在本线程停止运动并退出。")
        self._refresh_authorization_button()

    def _on_competition_progress(self, event: object) -> None:
        value = event if isinstance(event, dict) else {"phase": "event", "message": str(event)}
        phase = str(value.get("phase", "event")); message = str(value.get("message", ""))
        line = f"[{phase}] {message}"
        self.monitor_status.setText(line)
        if phase.startswith("manual_text") or phase.startswith("countdown"):
            self.text_control_status.setText(message)
        if phase == "visual_result":
            self._show_visual_result(value, self.monitor_visual_status, self.monitor_visual_image)
        self.monitor_log.appendPlainText(f"{datetime.now().isoformat(timespec='milliseconds')}  {line}")
        self._log(line)

    def _on_competition_finished(self) -> None:
        self.monitor_status.setText("本场双任务卡流程完成")
        self.session_status.setText("本场比赛已完成")
        self._log("正式双任务卡流程完成。")

    def _on_competition_failed(self, reason: str) -> None:
        self.monitor_status.setText(f"流程失败并停止：{reason}")
        self.session_status.setText("授权已撤销；故障排除后必须重新检查和授权")
        self._log(f"正式流程失败：{reason}")

    def _cleanup_competition_worker(self) -> None:
        thread = self.competition_thread
        worker = self.competition_worker
        self.competition_thread = None
        self.competition_worker = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
        self._refresh_text_controls()
        self._refresh_authorization_button()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.voice_thread is not None and self.voice_thread.isRunning():
            if not self.voice_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "语音盒子正在录音、识别或播报；请等待本次请求完成后再关闭窗口。"); return
        if self.point_move_thread is not None and self.point_move_thread.isRunning():
            if self.point_move_worker is not None:
                self.point_move_worker.request_stop()
            if not self.point_move_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在安全停止", "点位移动Worker尚未退出，窗口不会强制关闭。请保持急停可用并等待停止完成。"); return
        if self.contact_capture_thread is not None and self.contact_capture_thread.isRunning():
            if not self.contact_capture_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "红色抓取/放置基准采集Worker尚未退出，窗口不会强制关闭。"); return
        if self.mvs_read_thread is not None and self.mvs_read_thread.isRunning():
            if not self.mvs_read_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "MVS参数只读Worker尚未退出，窗口不会强制关闭。"); return
        if self.point_capture_thread is not None and self.point_capture_thread.isRunning():
            if not self.point_capture_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "点位只读采集Worker尚未退出，窗口不会强制关闭。"); return
        if self.io_thread is not None and self.io_thread.isRunning():
            if not self.io_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "真实 IO Worker尚未退出，窗口不会强制关闭。"); return
        if self.readiness_thread is not None and self.readiness_thread.isRunning():
            if not self.readiness_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "机械臂只读检查Worker尚未退出，窗口不会强制关闭。"); return
        if self.profile_thread is not None and self.profile_thread.isRunning():
            if not self.profile_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "MVS参数或任务卡大模型测试Worker尚未退出，窗口不会强制关闭。"); return
        if self.detector_thread is not None and self.detector_thread.isRunning():
            if not self.detector_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "颜色验证Worker尚未退出，窗口不会强制关闭。"); return
        if self.calibration_thread is not None and self.calibration_thread.isRunning():
            if self.calibration_worker is not None: self.calibration_worker.request_stop()
            if not self.calibration_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在安全停止", "九点Worker尚未退出，窗口不会强制关闭。"); return
        if self.validation_thread is not None and self.validation_thread.isRunning():
            if not self.validation_thread.wait(5000):
                event.ignore(); QMessageBox.warning(self, "仍在等待", "九点验证Worker尚未退出，窗口不会强制关闭。"); return
        thread = self.competition_thread
        if thread is not None and thread.isRunning():
            if self.competition_worker is not None:
                self.competition_worker.request_stop()
            if not thread.wait(5000):
                event.ignore()
                QMessageBox.warning(self, "仍在安全停止", "Robot Worker尚未退出，窗口不会强制关闭。请等待停止完成后再关闭。")
                return
        event.accept()


def run_app() -> int:
    load_all()
    app = QApplication.instance() or QApplication([])
    window = CompetitionWindow(); window.show()
    import os
    if os.environ.get("COMPETITION_STARTUP_CHECK_ONLY") == "1":
        app.processEvents()
        if window.authorize_button.isEnabled():
            raise RuntimeError("未配置硬件时比赛授权按钮不应启用。")
        window.close()
        return 0
    return app.exec()

