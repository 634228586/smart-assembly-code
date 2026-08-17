from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .robot_gateway import AuboRealGateway, MotionPermit


class ConfiguredRobotPort:
    """把已批准配置和当前会话绑定到装夹执行器使用的窄接口。"""

    def __init__(self, gateway: AuboRealGateway, *, session_id: str, config_fingerprint: str, motion: dict[str, Any], suction_io: dict[str, Any], stop_event: threading.Event | None = None, fingerprint_provider: Callable[[], str] | None = None, on_command: Callable[[str], None] | None = None) -> None:
        self.gateway = gateway; self.session_id = session_id; self.fingerprint = config_fingerprint
        self.motion = motion; self.suction_io = suction_io; self.on_command = on_command or (lambda message: None)
        self.stop_event = stop_event or threading.Event()
        self.fingerprint_provider = fingerprint_provider or (lambda: self.fingerprint)

    def _permit(self) -> MotionPermit:
        if self.fingerprint_provider() != self.fingerprint:
            self.stop_event.set()
            raise RuntimeError("config/real 在比赛会话中发生改变，已阻止下一条硬件命令。")
        return MotionPermit(self.session_id, time.monotonic(), self.fingerprint)

    def current_tcp_pose(self) -> tuple[float, ...]:
        return self.gateway.current_tcp_pose()

    def pose_trans(self, base_pose, tool_delta) -> tuple[float, ...]:
        return self.gateway.pose_trans(base_pose, tool_delta)

    def move_joint(self, target) -> None:
        self.on_command("moveJoint")
        self.gateway.move_joint_and_wait(target, self.motion["limits"], self._permit(), self.fingerprint, should_stop=self.stop_event.is_set)

    def move_line(self, target) -> None:
        self.on_command("moveLine")
        self.gateway.move_line_and_wait(target, self.motion["limits"], self._permit(), self.fingerprint, should_stop=self.stop_event.is_set)

    def set_suction(self, enabled: bool) -> None:
        self.on_command("suction_on" if enabled else "suction_off")
        self.gateway.set_suction(enabled, self.suction_io, self._permit(), self.fingerprint, should_stop=self.stop_event.is_set)

    def request_stop(self) -> None:
        """只设置线程安全标志；实际 SDK stop 由 Worker 的等待循环调用。"""

        self.stop_event.set()
