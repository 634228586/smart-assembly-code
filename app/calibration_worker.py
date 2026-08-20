from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .config import endpoints, load_all
from .integrity import current_runtime_fingerprint
from .nine_point import build_grid
from .robot_gateway import AuboRealGateway, MotionPermit
from .vision_client import RealVisionClient
from vision.mvs_camera import MvsCamera


def calibration_motion_limits(motion: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """生成九点中心/网格运动限制，并在连接硬件前校验速度上限。"""

    nine = motion["nine_point"]
    center_limits = dict(motion["limits"])
    grid_limits = dict(center_limits)
    grid_limits["speed_fraction"] = float(nine["speed_fraction"])
    grid_limits["linear_acceleration_m_s2"] = float(nine["linear_acceleration_m_s2"])
    grid_limits["linear_velocity_m_s"] = float(nine["linear_velocity_m_s"])
    grid_limits["tcp_position_tolerance_m"] = min(float(grid_limits["tcp_position_tolerance_m"]), 0.0005)
    if float(grid_limits["maximum_authorized_speed_fraction"]) < float(grid_limits["speed_fraction"]):
        raise RuntimeError("机器人已批准最大速度比例低于九点网格运动速度。")
    return center_limits, grid_limits


class CurrentMvsParametersWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        camera: MvsCamera | None = None
        try:
            configs = load_all()
            camera = MvsCamera()
            device = camera.open_first_available()
            parameters = camera.read_current_parameters()
            self.finished.emit({
                "model": device.model,
                "transport": device.transport,
                "parameters": parameters,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass


class RobotReadinessWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            configs = load_all(); fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            gateway.connect_readonly()
            snapshot = gateway.assert_runtime_start_gate(
                MotionPermit("readonly-calibration-readiness", time.monotonic(), fingerprint), fingerprint
            )
            self.finished.emit({
                "robot_name": snapshot.robot_name,
                "robot_mode": snapshot.robot_mode,
                "safety_mode": snapshot.safety_mode,
                "exec_id": snapshot.exec_id,
                "runtime_state": snapshot.runtime_state,
                "tcp_offset": list(snapshot.tcp_offset),
                "active_tcp": configs["robot"]["active_tcp"]["name"],
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()


class CalibrationWorker(QObject):
    progress = Signal(object)
    confirmation_needed = Signal(object)
    candidate_ready = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, *, scene: str, automatic: bool = False) -> None:
        super().__init__()
        self.scene = scene
        self.automatic = bool(automatic)
        self.stop_event = threading.Event()
        self.confirm_event = threading.Event()
        self._gateway: AuboRealGateway | None = None
        self._vision: RealVisionClient | None = None
        self._session_id = ""

    def accept_current_step(self) -> None:
        self.confirm_event.set()

    def request_stop(self) -> None:
        self.stop_event.set(); self.confirm_event.set()

    def _wait_confirmation(self, payload: dict[str, Any]) -> None:
        if self.automatic:
            return
        self.confirm_event.clear(); self.confirmation_needed.emit(payload)
        while not self.confirm_event.wait(0.1):
            if self.stop_event.is_set():
                raise RuntimeError("收到人工停止请求。")
        if self.stop_event.is_set():
            raise RuntimeError("收到人工停止请求。")

    def _permit(self, fingerprint: str) -> MotionPermit:
        return MotionPermit(f"calibration:{self._session_id}", time.monotonic(), fingerprint)

    @Slot()
    def run(self) -> None:
        try:
            if self.scene not in {"blocks", "trays"}:
                raise RuntimeError("九点场景无效。")
            configs = load_all(); motion = configs["motion"]; nine = motion["nine_point"]; scene_cfg = nine[self.scene]
            configured = endpoints(configs["endpoints"])
            if "robot_rpc" not in configured or "vision_service" not in configured:
                raise RuntimeError("机器人或视觉端点尚未配置。")
            fingerprint = current_runtime_fingerprint()
            center_limits, grid_limits = calibration_motion_limits(motion)
            self._session_id = f"nine-{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"]); self._gateway = gateway
            snapshot = gateway.connect_readonly()
            self.progress.emit({"phase": "identity", "message": f"真实机器人身份通过：{snapshot.robot_name}"})
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"], fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
                visual_result_callback=lambda payload: self.progress.emit({**payload, "phase": "visual_result"}),
            ); self._vision = vision
            health = vision.health()
            stale_session = health.get("calibration_session")
            if isinstance(stale_session, str) and stale_session:
                vision.calibration_abort(
                    request_id=f"{self._session_id}-discard-stale",
                    session_id=stale_session,
                )
                self.progress.emit({
                    "phase": "stale_reset",
                    "message": f"已废弃旧的未完成九点会话 {stale_session}；本次从第1点重新开始。",
                })
            self.progress.emit({"phase": "identity", "message": "真实MVS服务和相机身份通过。"})

            runtime_started = gateway.start_runtime_for_maintenance(self._permit(fingerprint), fingerprint)
            self.progress.emit({
                "phase": "runtime_start",
                "message": "已自动启动 RuntimeMachine（未调用 runProgram）。" if runtime_started else "RuntimeMachine 已处于 Running。",
            })

            center_speed_percent = float(center_limits["speed_fraction"]) * 100.0
            self.progress.emit({"phase": "center_move", "message": f"{center_speed_percent:g}%直接 MoveJoint 到 {self.scene}拍照点；不使用路线验收。"})
            gateway.move_joint_maintenance_and_wait(motion["points"][f"{self.scene}_photo"], center_limits, self._permit(fingerprint), fingerprint, should_stop=self.stop_event.is_set)
            center_pose = gateway.current_tcp_pose()
            begin = vision.calibration_begin(
                request_id=f"{self._session_id}-begin", session_id=self._session_id,
                scene=self.scene, target_color=scene_cfg["target_color"], photo_point=f"{self.scene}_photo",
                robot_serial=configs["robot"]["identity"]["binding_id"],
                step_x_mm=scene_cfg["step_x_mm"], step_y_mm=scene_cfg["step_y_mm"],
            )
            replaced = begin.get("replaced_session_id")
            reset_note = f"；已替换旧会话 {replaced}" if isinstance(replaced, str) and replaced else ""
            self.progress.emit({"phase": "precheck", "message": f"中心预检通过，唯一目标清晰；本次从第1点重新采集{reset_note}。", **begin})
            grid = build_grid(scene_cfg["step_x_mm"], scene_cfg["step_y_mm"])
            for point in grid:
                target = gateway.pose_trans(center_pose, (point.camera_x_mm / 1000.0, point.camera_y_mm / 1000.0, 0, 0, 0, 0))
                self._wait_confirmation({
                    "kind": "move", "index": point.index,
                    "message": f"确认执行第{point.index}/9点：相机 X={point.camera_x_mm:.3f} mm，Y={point.camera_y_mm:.3f} mm",
                    "target_pose": target,
                })
                gateway.move_line_maintenance_and_wait(target, grid_limits, self._permit(fingerprint), fingerprint, should_stop=self.stop_event.is_set)
                if self.stop_event.wait(float(nine["settle_s"])):
                    raise RuntimeError("稳定等待期间收到人工停止。")
                actual = gateway.current_tcp_pose()
                relative = gateway.pose_trans_inv(center_pose, actual)
                response = vision.calibration_capture_point(
                    request_id=f"{self._session_id}-point-{point.index}", session_id=self._session_id,
                    index=point.index, actual_tcp_pose=list(actual),
                    tool_x_mm=-relative[0] * 1000.0, tool_y_mm=-relative[1] * 1000.0,
                )
                self.progress.emit({"phase": "point", "message": f"第{point.index}/9点真实新帧采集完成。", **response})
                self._wait_confirmation({
                    "kind": "sample", "index": point.index,
                    "message": f"检查第{point.index}/9点图像、中心和位姿；确认接受该样本。",
                    "sample": response.get("sample"),
                })
            self.progress.emit({"phase": "center_return", "message": f"九点采集完成，{center_speed_percent:g}%直接返回 {self.scene}拍照中心。"})
            gateway.move_joint_maintenance_and_wait(motion["points"][f"{self.scene}_photo"], center_limits, self._permit(fingerprint), fingerprint, should_stop=self.stop_event.is_set)
            finish = vision.calibration_finish(request_id=f"{self._session_id}-finish", session_id=self._session_id)
            self.candidate_ready.emit(finish)
            self.finished.emit()
        except Exception as exc:
            if self._vision is not None and self._session_id:
                try:
                    self._vision.calibration_abort(request_id=f"{self._session_id}-abort", session_id=self._session_id)
                except Exception:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if self._gateway is not None:
                self._gateway.disconnect()
                self._gateway = None


class CalibrationValidationWorker(QObject):
    finished = Signal(object)
    visual = Signal(object)
    failed = Signal(str)

    def __init__(self, *, session_id: str, scene: str, validation_kind: str) -> None:
        super().__init__()
        self.session_id = session_id; self.scene = scene; self.validation_kind = validation_kind

    @Slot()
    def run(self) -> None:
        gateway: AuboRealGateway | None = None
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            fingerprint = current_runtime_fingerprint()
            gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
            gateway.connect_readonly()
            snapshot = gateway.assert_motion_gate(MotionPermit(f"validation:{self.session_id}", time.monotonic(), fingerprint), fingerprint)
            expected = tuple(float(value) for value in configs["motion"]["points"][f"{self.scene}_photo"])
            tolerance = float(configs["motion"]["limits"]["joint_tolerance_rad"])
            if max(abs(actual - target) for actual, target in zip(snapshot.joint_positions, expected)) > tolerance:
                raise RuntimeError(f"方向验证时机器人不在 {self.scene}中心拍照点；禁止用错误姿态拍照。")
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
                visual_result_callback=self.visual.emit,
            )
            vision.health()
            result = vision.calibration_validate_capture(
                request_id=f"{self.session_id}-validate-{self.validation_kind}",
                session_id=self.session_id, scene=self.scene, validation_kind=self.validation_kind,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if gateway is not None:
                gateway.disconnect()


class DetectorValidationWorker(QObject):
    finished = Signal(object)
    visual = Signal(object)
    failed = Signal(str)

    def __init__(self, *, scene: str) -> None:
        super().__init__(); self.scene = scene

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            session_id = f"detector-{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
                visual_result_callback=self.visual.emit,
            )
            vision.health()
            result = vision.validate_detector(request_id=f"{session_id}-capture", session_id=session_id, scene=self.scene)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DetectorAreaEstimateWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, scene: str) -> None:
        super().__init__(); self.scene = scene

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            session_id = f"area-{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            )
            vision.health()
            self.finished.emit(vision.estimate_detector_area(
                request_id=f"{session_id}-capture", session_id=session_id, scene=self.scene,
            ))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ManualSceneCaptureWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, scene: str) -> None:
        super().__init__(); self.scene = scene

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            session_id = f"manual-{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            )
            vision.health()
            self.finished.emit(vision.capture_manual_scene(
                request_id=f"{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}",
                session_id=session_id,
                scene=self.scene,
            ))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ManualSceneRecognitionWorker(QObject):
    """Capture and recognize one new scene frame; never moves or changes approvals."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, scene: str) -> None:
        super().__init__(); self.scene = scene

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            session_id = f"manual-recognize-{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            )
            vision.health()
            self.finished.emit(vision.recognize_manual_scene(
                request_id=f"{self.scene}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}",
                session_id=session_id,
                scene=self.scene,
            ))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ProfileValidationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            configs = load_all(); configured = endpoints(configs["endpoints"])
            session_id = f"profiles-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            vision = RealVisionClient(
                configured["vision_service"],
                active_tcp=configs["robot"]["active_tcp"]["name"],
                fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            )
            vision.health()
            results = [vision.validate_profile(request_id=f"{session_id}-{name}", session_id=session_id, profile=name) for name in ("task_card", "blocks", "trays")]
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
