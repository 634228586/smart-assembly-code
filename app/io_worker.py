from __future__ import annotations

import time
from typing import Literal

from PySide6.QtCore import QObject, Signal, Slot

from .config import load_all
from .integrity import current_runtime_fingerprint
from .robot_gateway import AuboRealGateway, MotionPermit, RobotGatewayError


IoOperation = Literal["toggle_aperture", "suction_on", "suction_off"]


class ManualIoWorker(QObject):
    """独占Worker：独立于比赛门控，保留身份与基础安全状态后单步写IO。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: IoOperation) -> None:
        super().__init__()
        self.operation = operation

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            configs = load_all()
            fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            snapshot = gateway.connect_readonly()
            permit = MotionPermit(f"manual-io:{self.operation}", time.monotonic(), fingerprint)
            gateway.assert_io_gate(permit, fingerprint)
            io_config = configs["suction_io"]
            if self.operation == "toggle_aperture":
                aperture = io_config.get("aperture", {})
                if aperture.get("output_type") != "standard_digital_output":
                    raise RobotGatewayError("光圈输出类型未配置为标准数字输出。")
                enabled = gateway.toggle_standard_digital_output(
                    index=aperture.get("output_index"),
                    on_level=aperture.get("on_level"),
                    off_level=aperture.get("off_level"),
                    permit=permit,
                    fingerprint=fingerprint,
                    label="光圈 DO",
                    confirmation_timeout_s=float(io_config.get("confirmation_timeout_s", 2.0)),
                )
                result = {"device": "光圈", "index": aperture["output_index"], "enabled": enabled}
            else:
                enabled = self.operation == "suction_on"
                gateway.set_suction_maintenance(enabled, io_config, permit, fingerprint)
                result = {
                    "device": "吸盘", "index": io_config["output_index"],
                    "enable_index": io_config["enable_output_index"], "enabled": enabled,
                }
            result["robot_name"] = snapshot.robot_name
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()
