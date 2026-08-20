from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .assembly import AssemblyExecutor
from .config import endpoints, load_all
from .integrity import current_runtime_fingerprint
from .real_ports import ConfiguredRobotPort
from .robot_gateway import AuboRealGateway
from .runtime import RuntimeBuildError, _calibration_id
from .vision_client import RealVisionClient


class DirectAssemblyWorker(QObject):
    """执行一组真实抓放；跳过任务卡/语音，但保留机器人、视觉和标定门控。"""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, block_color: str, tray_color: str) -> None:
        super().__init__()
        self.block_color = block_color
        self.tray_color = tray_color
        self.stop_event = threading.Event()
        self.gateway: AuboRealGateway | None = None
        self.executor: AssemblyExecutor | None = None

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all()
            configured = endpoints(configs["endpoints"])
            for role in ("robot_rpc", "vision_service"):
                if role not in configured:
                    raise RuntimeBuildError(f"端点 {role} 尚未配置。")
            calibration_ids = {scene: _calibration_id(scene) for scene in ("blocks", "trays")}
            fingerprint = current_runtime_fingerprint()
            session_id = datetime.now().strftime("direct-one-%Y%m%d-%H%M%S-%f")
            self.gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            snapshot = self.gateway.connect_readonly()
            self.progress.emit({"phase": "robot_identity", "message": f"真实机器人只读身份通过：{snapshot.robot_name}"})

            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                calibration_ids=calibration_ids,
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
                visual_result_callback=lambda payload: self.progress.emit({**payload, "phase": "visual_result"}),
            )
            vision.health()
            self.progress.emit({"phase": "vision_identity", "message": "真实 MVS视觉服务和两套标定已通过。"})

            robot = ConfiguredRobotPort(
                self.gateway,
                session_id=session_id,
                config_fingerprint=fingerprint,
                motion=configs["motion"],
                suction_io=configs["suction_io"],
                stop_event=self.stop_event,
                fingerprint_provider=current_runtime_fingerprint,
                on_command=lambda command: self.progress.emit({"phase": "robot_command", "message": command}),
            )
            self.executor = AssemblyExecutor(
                robot,
                vision,
                session_id=session_id,
                points=configs["motion"]["points"],
                reference_anchors=configs["motion"]["reference_anchors"],
                progress=self.progress.emit,
                stop_event=self.stop_event,
            )
            result = self.executor.run_single(block_color=self.block_color, tray_color=self.tray_color)
            self.finished.emit({
                "block_color": self.block_color,
                "tray_color": self.tray_color,
                "completed_cycles": result.completed_cycles,
                "suction_may_be_on": result.suction_may_be_on,
            })
        except Exception as exc:
            suction = self.executor.suction_may_be_on if self.executor is not None else False
            suffix = "；吸盘可能仍在吸取，请人工确认后处理" if suction else ""
            self.failed.emit(f"{type(exc).__name__}: {exc}{suffix}")
        finally:
            self.executor = None
            if self.gateway is not None:
                self.gateway.disconnect()
                self.gateway = None

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.executor is not None:
            self.executor.request_stop()
