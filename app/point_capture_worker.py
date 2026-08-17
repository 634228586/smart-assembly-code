from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, Signal, Slot

from .config import load_all
from .integrity import current_runtime_fingerprint
from .paths import REAL_CONFIG_DIR
from .robot_config_editor import COLORS, POINT_KEYS, POINT_LABELS, REFERENCE_LABELS, save_reference_anchor, save_single_joint_point
from .robot_gateway import AuboRealGateway, MotionPermit


def maintenance_point_limits(motion_config: dict) -> dict:
    """Return an isolated limits copy for the manual point-move button."""

    limits = dict(motion_config["limits"])
    maintenance = motion_config.get("maintenance_point")
    if not isinstance(maintenance, dict):
        raise RuntimeError("维护单点速度配置缺失。")
    speed = float(maintenance.get("speed_fraction"))
    maximum = float(maintenance.get("maximum_authorized_speed_fraction"))
    if not 0 < speed <= maximum <= 1:
        raise RuntimeError("维护单点速度或批准上限无效。")
    limits["speed_fraction"] = speed
    limits["maximum_authorized_speed_fraction"] = maximum
    return limits


class RobotPointCaptureWorker(QObject):
    """读取唯一机器人的当前关节角并直接保存为一个指定正式点位；绝不运动。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, point_key: str) -> None:
        super().__init__()
        self.point_key = point_key

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            if self.point_key not in POINT_KEYS:
                raise RuntimeError("未知关节点。")
            configs = load_all(); fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            gateway.connect_readonly()
            snapshot = gateway.assert_maintenance_gate(
                MotionPermit(f"capture-point:{self.point_key}", time.monotonic(), fingerprint), fingerprint
            )
            saved = save_single_joint_point(
                REAL_CONFIG_DIR / "motion.json", point_key=self.point_key, joint_positions=snapshot.joint_positions
            )
            self.finished.emit({
                "point_key": self.point_key,
                "point_label": POINT_LABELS[self.point_key],
                "robot_name": snapshot.robot_name,
                "joint_positions_rad": saved,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()


class RobotPointMoveWorker(QObject):
    """低速移动唯一机器人到一个已保存关节点，并等待真实到位。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, point_key: str) -> None:
        super().__init__()
        self.point_key = point_key
        self.stop_event = threading.Event()

    @Slot()
    def request_stop(self) -> None:
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            if self.point_key not in POINT_KEYS:
                raise RuntimeError("未知关节点。")
            configs = load_all()
            target = configs["motion"]["points"].get(self.point_key)
            if not isinstance(target, (list, tuple)) or len(target) != 6:
                raise RuntimeError(f"{POINT_LABELS[self.point_key]}尚未保存完整六轴关节角。")
            point_limits = maintenance_point_limits(configs["motion"])
            fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            gateway.connect_readonly()
            permit = MotionPermit(
                f"manual-move-point:{self.point_key}", time.monotonic(), fingerprint
            )
            gateway.move_joint_maintenance_and_wait(
                target,
                point_limits,
                permit,
                fingerprint,
                should_stop=self.stop_event.is_set,
            )
            snapshot = gateway.snapshot()
            self.finished.emit({
                "point_key": self.point_key,
                "point_label": POINT_LABELS[self.point_key],
                "robot_name": snapshot.robot_name,
                "joint_positions_rad": list(snapshot.joint_positions),
                "speed_fraction": float(point_limits["speed_fraction"]),
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()


class RobotReferenceAnchorCaptureWorker(QObject):
    """Read and save one colour-specific operator-aligned TCP pose; never moves."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, scene: str, color: str = "红") -> None:
        super().__init__(); self.scene = scene; self.color = color

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            if self.scene not in REFERENCE_LABELS:
                raise RuntimeError("未知基准抓取/放置场景。")
            if self.color not in COLORS:
                raise RuntimeError("未知基准点颜色。")
            configs = load_all(); fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            gateway.connect_readonly()
            snapshot = gateway.assert_maintenance_gate(
                MotionPermit(f"capture-reference-anchor:{self.scene}:{self.color}", time.monotonic(), fingerprint), fingerprint
            )
            saved = save_reference_anchor(
                REAL_CONFIG_DIR / "motion.json", scene=self.scene, color=self.color, tcp_pose=snapshot.tcp_pose
            )
            self.finished.emit({
                "scene": self.scene, "color": self.color, "label": f"{REFERENCE_LABELS[self.scene]}({self.color})",
                "robot_name": snapshot.robot_name, "tcp_pose": saved,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()
