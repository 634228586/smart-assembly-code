from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socketserver
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.config import Endpoint, endpoints, load_all, walk_unset
from app.nine_point import NinePointError, build_candidate, build_grid, fit_pixel_to_tool, write_candidate
from app.paths import REAL_CALIBRATION_DIR, SESSION_DIR, ensure_runtime_directories
from vision.mvs_camera import MvsCamera, MvsCameraError, sdk_files_available
from vision.workspace_localizer import ApprovedCalibration, COLORS, WorkspaceRecognitionError, build_detection_diagnostic, detect_calibration_color_pixel, detect_color_pixel, estimate_six_color_area_range, load_approved_calibration, locate_calibration_color, locate_colors


SERVICE_NAME = "real_mvs_vision"
PROTOCOL_VERSION = 1
SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


def _camera_start_missing(camera_config: dict[str, Any]) -> list[str]:
    required: dict[str, Any] = {
        "serial_number": camera_config.get("serial_number"),
        "sdk_family": camera_config.get("sdk_family"),
        "mounting": camera_config.get("mounting"),
        "fresh_frame_max_age_ms": camera_config.get("fresh_frame_max_age_ms"),
    }
    profiles = camera_config.get("profiles")
    if not isinstance(profiles, dict):
        required["profiles"] = "UNSET"
    else:
        for name in ("task_card", "blocks", "trays"):
            profile = profiles.get(name)
            if not isinstance(profile, dict):
                required[f"profiles.{name}"] = "UNSET"
                continue
            required[f"profiles.{name}"] = {
                key: profile.get(key) for key in ("exposure_us", "gain", "white_balance", "roi", "trigger_mode")
            }
    return list(walk_unset(required))


class VisionServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class VisionCalibrationSession:
    session_id: str
    scene: str
    target_color: str
    photo_point: str
    robot_serial: str
    active_tcp: str
    step_x_mm: float
    step_y_mm: float
    image_width: int
    image_height: int
    samples: list[dict[str, Any]] = field(default_factory=list)


class RealMvsVisionService:
    def __init__(self, *, camera: Any, camera_config: dict[str, Any], endpoint: Endpoint, calibration_root: Path = REAL_CALIBRATION_DIR, session_root: Path = SESSION_DIR) -> None:
        if endpoint.expected_service != SERVICE_NAME or endpoint.host not in {"127.0.0.1", "localhost", "::1"}:
            raise VisionServiceError("ENDPOINT_INVALID", "真实 MVS视觉服务只允许绑定配置的本机端点。")
        missing = _camera_start_missing(camera_config)
        if missing:
            raise VisionServiceError("CAMERA_CONFIG_INCOMPLETE", f"相机配置尚未补齐：{', '.join(missing)}")
        if camera_config.get("sdk_family") != "hikrobot_mvs" or camera_config.get("mounting") != "eye_in_hand":
            raise VisionServiceError("CAMERA_CONFIG_INVALID", "相机 SDK家族或眼在手安装方式不匹配。")
        self.camera = camera
        self.camera_config = camera_config
        self.endpoint = endpoint
        self.calibration_root = calibration_root.resolve()
        self.session_root = session_root.resolve()
        self.configured_serial = str(camera_config["serial_number"])
        self.serial = self.configured_serial
        self.profiles = camera_config["profiles"]
        self._lock = threading.Lock()
        self._calibration: VisionCalibrationSession | None = None
        self._candidates: dict[tuple[str, str], Path] = {}
        self.device = self.camera.open_exact_serial(self.serial)
        # 以实际打开的唯一设备为本次服务身份；允许配置文件暂留旧序列号。
        self.serial = self.device.serial

    @classmethod
    def from_real_config(cls) -> "RealMvsVisionService":
        configs = load_all()
        configured = endpoints(configs["endpoints"])
        endpoint = configured.get("vision_service")
        if endpoint is None:
            raise VisionServiceError("ENDPOINT_UNSET", "vision_service端点尚未配置。")
        camera_config = configs["camera"]
        missing = _camera_start_missing(camera_config)
        if missing:
            raise VisionServiceError("CAMERA_CONFIG_INCOMPLETE", f"相机配置尚未补齐：{', '.join(missing)}")
        return cls(camera=MvsCamera(), camera_config=camera_config, endpoint=endpoint)

    def close(self) -> None:
        self.camera.close()

    def handle(self, request: Any) -> dict[str, Any]:
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict):
                raise VisionServiceError("INVALID_REQUEST", "请求必须是 JSON对象。")
            request_type = request.get("type")
            if request.get("protocol_version") != PROTOCOL_VERSION:
                raise VisionServiceError("PROTOCOL_MISMATCH", "视觉协议版本不匹配。")
            with self._lock:
                if request_type == "health_request":
                    return self._health(request)
                if request_type == "calibration_begin":
                    return self._calibration_begin(request)
                if request_type == "calibration_capture_point":
                    return self._calibration_capture_point(request)
                if request_type == "calibration_finish":
                    return self._calibration_finish(request)
                if request_type == "calibration_abort":
                    return self._calibration_abort(request)
                if request_type == "calibration_validate_capture":
                    return self._calibration_validate_capture(request)
                if self._calibration is not None:
                    raise VisionServiceError("CALIBRATION_BUSY", "识别失败：真实九点标定会话进行中，装夹拍照已锁定。")
                if request_type == "profile_validate":
                    return self._profile_validate(request)
                if request_type == "detector_validate":
                    return self._detector_validate(request)
                if request_type == "detector_estimate_area":
                    return self._detector_estimate_area(request)
                if request_type == "manual_scene_capture":
                    return self._manual_scene_capture(request)
                if request_type == "manual_scene_recognize":
                    return self._manual_scene_recognize(request)
                if request_type == "capture_frame":
                    return self._capture_task_card(request)
                if request_type == "capture_and_locate":
                    return self._capture_and_locate(request)
            raise VisionServiceError("UNKNOWN_REQUEST", "未知视觉请求类型。")
        except (VisionServiceError, MvsCameraError, WorkspaceRecognitionError, NinePointError) as exc:
            response = {
                "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
                "type": "error_result", "success": False,
                "request_id": request_id if isinstance(request_id, str) else "",
                "error_code": getattr(exc, "code", type(exc).__name__),
                "message": str(exc) if str(exc).startswith("识别失败") else f"识别失败：{exc}",
            }
            if isinstance(exc, VisionServiceError):
                response.update(exc.details)
            return response
        except Exception as exc:
            return {
                "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
                "type": "error_result", "success": False,
                "request_id": request_id if isinstance(request_id, str) else "",
                "error_code": "INTERNAL_ERROR", "message": f"识别失败：视觉服务内部错误：{type(exc).__name__}",
            }

    def _health(self, request: dict[str, Any]) -> dict[str, Any]:
        if set(request) != {"type", "protocol_version", "expected_service"} or request.get("expected_service") != SERVICE_NAME:
            raise VisionServiceError("HEALTH_IDENTITY_MISMATCH", "健康检查的服务身份字段不匹配。")
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "status": "ready", "camera_serial": self.serial,
            "camera_model": self.device.model, "camera_transport": self.device.transport,
            "source": "mvs", "mounting": "eye_in_hand",
            "calibration_session": self._calibration.session_id if self._calibration is not None else None,
        }

    def _calibration_begin(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "type", "protocol_version", "request_id", "session_id", "scene", "profile",
            "target_color", "photo_point", "camera_serial", "robot_serial", "active_tcp",
            "step_x_mm", "step_y_mm",
        }
        if set(request) != required:
            raise VisionServiceError("CALIBRATION_BEGIN_INVALID", "九点开始请求字段不完整或包含未知字段。")
        request_id, session_id = self._ids(request)
        self._assert_serial(request)
        scene = request.get("scene")
        if scene not in {"blocks", "trays"} or request.get("profile") != scene or request.get("photo_point") != f"{scene}_photo":
            raise VisionServiceError("CALIBRATION_SCENE_INVALID", "九点场景、profile或拍照点不匹配。")
        grid = build_grid(request.get("step_x_mm"), request.get("step_y_mm"))
        target_color = str(request.get("target_color", ""))
        replaced_session_id = self._calibration.session_id if self._calibration is not None else None
        # “开始/重新生成”永远创建本次全新九点；旧的未完成内存会话和旧候选不能干扰。
        self._calibration = None
        self._candidates = {
            key: path for key, path in self._candidates.items() if key[1] != scene
        }
        capture = self.camera.capture(self.profiles[scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "九点预检采集参数未成功写入。")
        height, width = capture.image_bgr.shape[:2]
        captured_at = datetime.now(timezone.utc).isoformat()
        raw_path, image_path, summary = self._save_detection_images(
            session_id, scene, "precheck", capture.image_bgr, colors=(target_color,),
            calibration_mode=True, storage_scene=f"calibration/{scene}",
        )
        try:
            detection = detect_calibration_color_pixel(capture.image_bgr, profile=self.profiles[scene], color=target_color)
        except WorkspaceRecognitionError as exc:
            raise VisionServiceError(
                exc.code, str(exc), details=self._diagnostic_details(scene, raw_path, image_path, summary),
            ) from exc
        self._calibration = VisionCalibrationSession(
            session_id=session_id, scene=scene, target_color=target_color,
            photo_point=str(request["photo_point"]), robot_serial=str(request["robot_serial"]),
            active_tcp=str(request["active_tcp"]), step_x_mm=float(request["step_x_mm"]),
            step_y_mm=float(request["step_y_mm"]), image_width=width, image_height=height,
        )
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "calibration_begin_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "camera_serial": self.serial, "captured_at": captured_at,
            "replaced_session_id": replaced_session_id,
            "frame_number": capture.frame_number, "image_width": width, "image_height": height,
            "image_path": str(image_path), "raw_image_path": str(raw_path),
            "annotated_image_path": str(image_path), "detection_summary": summary, "detection": detection,
            "grid": [{"index": point.index, "camera_x_mm": point.camera_x_mm, "camera_y_mm": point.camera_y_mm} for point in grid],
        }

    def _detector_validate(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "scene", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("DETECTOR_VALIDATE_INVALID", "颜色参数验证请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        scene = request.get("scene")
        if scene not in {"blocks", "trays"}:
            raise VisionServiceError("DETECTOR_VALIDATE_INVALID", "颜色参数验证场景无效。")
        capture = self.camera.capture(self.profiles[scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "颜色验证采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        colors = ("红", "橙", "黄", "绿", "蓝", "紫")
        raw_path, path, summary = self._save_detection_images(
            session_id, scene, request_id, capture.image_bgr, colors=colors,
            storage_scene=f"{scene}-detector-validation",
        )
        try:
            detections = [detect_color_pixel(capture.image_bgr, profile=self.profiles[scene], color=color, require_approved=False) for color in colors]
        except WorkspaceRecognitionError as exc:
            raise VisionServiceError(
                exc.code, str(exc), details=self._diagnostic_details(scene, raw_path, path, summary),
            ) from exc
        detector = self.profiles[scene]["detector"]
        detector_sha256 = hashlib.sha256(json.dumps(detector, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "detector_validate_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "camera_serial": self.serial, "captured_at": captured_at,
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "image_path": str(raw_path),
            "annotated_image_path": str(path), "detection_summary": summary,
            "detections": detections, "detector_sha256": detector_sha256,
        }

    def _detector_estimate_area(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "scene", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("DETECTOR_ESTIMATE_INVALID", "面积自动估算请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        scene = request.get("scene")
        if scene not in {"blocks", "trays"}:
            raise VisionServiceError("DETECTOR_ESTIMATE_INVALID", "面积自动估算场景无效。")
        capture = self.camera.capture(self.profiles[scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "面积自动估算时采集参数未成功写入。")
        estimate = estimate_six_color_area_range(capture.image_bgr, profile=self.profiles[scene])
        captured_at = datetime.now(timezone.utc).isoformat()
        path = self._save_image(session_id, f"{scene}-detector-area-estimate", request_id, capture.image_bgr)
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "detector_estimate_area_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "camera_serial": self.serial, "captured_at": captured_at,
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "image_path": str(path), **estimate,
        }

    def _manual_scene_capture(self, request: dict[str, Any]) -> dict[str, Any]:
        """Capture one raw blocks/trays frame for offline detector tuning."""

        required = {"type", "protocol_version", "request_id", "session_id", "scene", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("MANUAL_CAPTURE_INVALID", "手动取图请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        scene = request.get("scene")
        if scene not in {"blocks", "trays"}:
            raise VisionServiceError("MANUAL_CAPTURE_INVALID", "手动取图场景必须是 blocks 或 trays。")
        capture = self.camera.capture(self.profiles[scene], require_approved=False)
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "手动取图时采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        image_path = self._save_image(session_id, f"manual-captures/{scene}", request_id, capture.image_bgr)
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "manual_scene_capture_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "camera_serial": self.serial, "captured_at": captured_at,
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "image_path": str(image_path),
            "parameters_applied": True, "configured_parameters": capture.configured_parameters,
        }

    def _manual_scene_recognize(self, request: dict[str, Any]) -> dict[str, Any]:
        """Capture one new frame and diagnose all six colors without changing approvals."""

        required = {
            "type", "protocol_version", "request_id", "session_id", "scene",
            "camera_serial", "active_tcp",
        }
        if set(request) != required:
            raise VisionServiceError("MANUAL_RECOGNIZE_INVALID", "手动拍照识别请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        scene = request.get("scene")
        if scene not in {"blocks", "trays"}:
            raise VisionServiceError("MANUAL_RECOGNIZE_INVALID", "手动识别场景必须是 blocks 或 trays。")
        active_tcp = request.get("active_tcp")
        if not isinstance(active_tcp, str) or not active_tcp.strip():
            raise VisionServiceError("MANUAL_RECOGNIZE_INVALID", "手动识别活动TCP无效。")

        capture = self.camera.capture(self.profiles[scene], require_approved=False)
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "手动识别时采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        colors = ("红", "橙", "黄", "绿", "蓝", "紫")
        raw_path, annotated_path, summary = self._save_detection_images(
            session_id, scene, request_id, capture.image_bgr, colors=colors,
            selection_policy="best_effort", storage_scene=f"manual-recognition/{scene}",
        )

        calibration: ApprovedCalibration | None = None
        calibration_message = "未找到可用九点标定，仅返回像素识别结果。"
        try:
            calibration_path = self._single_calibration_file(scene)
            calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not isinstance(calibration_raw, dict):
                raise WorkspaceRecognitionError("CALIBRATION_INVALID", "标定文件不是JSON对象。")
            calibration_id = calibration_raw.get("calibration_id")
            if not isinstance(calibration_id, str) or not calibration_id:
                raise WorkspaceRecognitionError("CALIBRATION_INVALID", "标定文件缺少 calibration_id。")
            calibration = load_approved_calibration(
                calibration_path, scene=scene, camera_serial=self.serial,
                active_tcp=active_tcp, calibration_id=calibration_id,
            )
            calibration_message = "已使用当前批准九点标定计算工具坐标偏移。"
        except (OSError, json.JSONDecodeError, VisionServiceError, WorkspaceRecognitionError) as exc:
            calibration_message = f"九点标定不可用，仅返回像素识别结果：{exc}"

        detections: list[dict[str, Any]] = []
        missing_colors: list[str] = []
        for color in colors:
            try:
                if calibration is None:
                    detections.append(detect_color_pixel(
                        capture.image_bgr, profile=self.profiles[scene], color=color,
                        require_approved=False, selection_policy="best_effort",
                    ))
                else:
                    detections.extend(locate_colors(
                        capture.image_bgr, profile=self.profiles[scene], calibration=calibration,
                        requested_color=color, selection_policy="best_effort",
                    ))
            except WorkspaceRecognitionError as exc:
                if exc.code == "TARGET_NOT_FOUND":
                    missing_colors.append(color)
                    continue
                raise VisionServiceError(
                    exc.code, str(exc),
                    details=self._diagnostic_details(scene, raw_path, annotated_path, summary),
                ) from exc

        detector = self.profiles[scene].get("detector", {})
        detector_sha256 = hashlib.sha256(
            json.dumps(detector, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "manual_scene_recognize_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "camera_serial": self.serial, "active_tcp": active_tcp,
            "captured_at": captured_at, "frame_number": capture.frame_number,
            "image_width": int(capture.image_bgr.shape[1]), "image_height": int(capture.image_bgr.shape[0]),
            "image_path": str(raw_path), "annotated_image_path": str(annotated_path),
            "detection_summary": summary, "configured_parameters": capture.configured_parameters,
            "detector_sha256": detector_sha256, "detections": detections,
            "missing_colors": missing_colors,
            "calibration_available": calibration is not None,
            "calibration_id": calibration.calibration_id if calibration is not None else None,
            "photo_point": calibration.photo_point if calibration is not None else None,
            "calibration_message": calibration_message,
        }

    def _profile_validate(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "profile", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("PROFILE_VALIDATE_INVALID", "采集参数写入测试请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        profile_name = request.get("profile")
        if profile_name not in {"task_card", "blocks", "trays"}:
            raise VisionServiceError("PROFILE_VALIDATE_INVALID", "采集参数写入测试 profile无效。")
        profile = self.profiles[profile_name]
        capture = self.camera.capture(profile, require_approved=False)
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", f"{profile_name}采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        image_path = self._save_image(session_id, f"{profile_name}-profile-validation", request_id, capture.image_bgr)
        profile_sha256 = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "profile_validate_result", "success": True,
            "request_id": request_id, "session_id": session_id, "profile": profile_name,
            "camera_serial": self.serial, "captured_at": captured_at,
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "image_path": str(image_path),
            "parameters_applied": True, "configured_parameters": capture.configured_parameters,
            "profile_sha256": profile_sha256,
        }

    def _calibration_capture_point(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "type", "protocol_version", "request_id", "session_id", "camera_serial",
            "index", "actual_tcp_pose", "tool_x_mm", "tool_y_mm",
        }
        if set(request) != required:
            raise VisionServiceError("CALIBRATION_POINT_INVALID", "九点采集请求字段不完整或包含未知字段。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        session = self._require_calibration(session_id)
        index = request.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != len(session.samples) + 1 or not 1 <= index <= 9:
            raise VisionServiceError("CALIBRATION_POINT_ORDER", f"九点点号无效，应为{len(session.samples) + 1}。")
        tcp_pose = request.get("actual_tcp_pose")
        if not isinstance(tcp_pose, list) or len(tcp_pose) != 6:
            raise VisionServiceError("CALIBRATION_POINT_INVALID", "实际 TCP pose必须包含六个数值。")
        try:
            normalized_pose = [float(value) for value in tcp_pose]
            tool_x_mm, tool_y_mm = float(request["tool_x_mm"]), float(request["tool_y_mm"])
        except (TypeError, ValueError) as exc:
            raise VisionServiceError("CALIBRATION_POINT_INVALID", "实际位姿或工具偏移不是数值。") from exc
        if not all(math.isfinite(value) for value in (*normalized_pose, tool_x_mm, tool_y_mm)):
            raise VisionServiceError("CALIBRATION_POINT_INVALID", "实际位姿或工具偏移包含非有限数值。")
        capture = self.camera.capture(self.profiles[session.scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "九点采集参数未成功写入。")
        if (capture.image_bgr.shape[1], capture.image_bgr.shape[0]) != (session.image_width, session.image_height):
            raise VisionServiceError("RESOLUTION_MISMATCH", "九点采集分辨率在会话中发生变化。")
        captured_at = datetime.now(timezone.utc).isoformat()
        raw_path, image_path, summary = self._save_detection_images(
            session_id, session.scene, f"point-{index}", capture.image_bgr,
            colors=(session.target_color,), calibration_mode=True,
            storage_scene=f"calibration/{session.scene}",
        )
        try:
            detection = detect_calibration_color_pixel(capture.image_bgr, profile=self.profiles[session.scene], color=session.target_color)
        except WorkspaceRecognitionError as exc:
            raise VisionServiceError(
                exc.code, str(exc), details=self._diagnostic_details(session.scene, raw_path, image_path, summary),
            ) from exc
        sample = {
            "index": index, "pixel_u": detection["pixel_u"], "pixel_v": detection["pixel_v"],
            "tool_x_mm": tool_x_mm, "tool_y_mm": tool_y_mm,
            "actual_tcp_pose": normalized_pose, "frame_number": capture.frame_number,
            "captured_at": captured_at, "image_path": str(image_path),
            "raw_image_path": str(raw_path), "annotated_image_path": str(image_path),
            "detection_summary": summary,
            "confidence": detection["confidence"], "r_image_deg": detection["r_image_deg"],
        }
        session.samples.append(sample)
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "calibration_point_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": session.scene,
            "camera_serial": self.serial, "sample": sample, "accepted_count": len(session.samples),
        }

    def _calibration_finish(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("CALIBRATION_FINISH_INVALID", "九点完成请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        session = self._require_calibration(session_id)
        if len(session.samples) != 9:
            raise VisionServiceError("CALIBRATION_INCOMPLETE", "九点样本未完整达到9个。")
        fit = fit_pixel_to_tool(session.samples)
        center_raw_path = Path(str(session.samples[4]["raw_image_path"])).resolve()
        try:
            center_image = cv2.imdecode(np.frombuffer(center_raw_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError as exc:
            raise VisionServiceError("CALIBRATION_REFERENCE_FAILED", "九点中心原图无法读取。") from exc
        if center_image is None:
            raise VisionServiceError("CALIBRATION_REFERENCE_FAILED", "九点中心原图无法解码。")
        reference_raw, reference_annotated, reference_summary = self._save_detection_images(
            session_id, session.scene, "six-color-reference", center_image,
            colors=COLORS, selection_policy="best_effort", storage_scene=f"calibration/{session.scene}",
        )
        references: dict[str, dict[str, float]] = {}
        for report in reference_summary.get("colors", []):
            selected = report.get("selected") if isinstance(report, dict) else None
            color = report.get("color") if isinstance(report, dict) else None
            if color not in COLORS or not isinstance(selected, dict):
                continue
            center = selected.get("center")
            if not isinstance(center, list) or len(center) != 2:
                continue
            references[str(color)] = {
                "pixel_u": float(center[0]), "pixel_v": float(center[1]),
                "r_image_deg": float(selected["angle_deg"]),
                "confidence": float(selected["confidence"]),
            }
        if set(references) != set(COLORS):
            missing = "、".join(color for color in COLORS if color not in references)
            raise VisionServiceError(
                "CALIBRATION_REFERENCE_INCOMPLETE",
                f"九点中心帧未完整识别六色基准框，缺少：{missing}；候选标定未生成。",
                details=self._diagnostic_details(session.scene, reference_raw, reference_annotated, reference_summary),
            )
        candidate = build_candidate(
            scene=session.scene, camera_serial=self.serial, robot_serial=session.robot_serial,
            active_tcp=session.active_tcp, photo_point=session.photo_point,
            image_width=session.image_width, image_height=session.image_height,
            target_color=session.target_color, step_x_mm=session.step_x_mm, step_y_mm=session.step_y_mm,
            samples=session.samples, fit=fit,
            reference_detections=references,
            reference_image_path=str(reference_raw),
            reference_annotated_image_path=str(reference_annotated),
        )
        path = (self.session_root / session_id / "calibration" / f"9point_{session.scene}_candidate.json").resolve()
        self._assert_session_path(path)
        write_candidate(path, candidate)
        self._candidates[(session_id, session.scene)] = path
        self._calibration = None
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "calibration_finish_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": session.scene,
            "camera_serial": self.serial, "candidate_path": str(path),
            "calibration_id": candidate["calibration_id"],
            "rms_error_mm": fit.rms_error_mm, "max_error_mm": fit.max_error_mm,
            "reference_detections": references,
            "reference_annotated_image_path": str(reference_annotated),
            "approved": False,
        }

    def _calibration_abort(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("CALIBRATION_ABORT_INVALID", "九点取消请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        self._require_calibration(session_id); self._calibration = None
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "calibration_abort_result", "success": True,
            "request_id": request_id, "session_id": session_id, "camera_serial": self.serial,
        }

    def _calibration_validate_capture(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "scene", "validation_kind", "camera_serial"}
        if set(request) != required:
            raise VisionServiceError("CALIBRATION_VALIDATE_INVALID", "九点方向验证请求字段无效。")
        request_id, session_id = self._ids(request); self._assert_serial(request)
        scene = request.get("scene"); kind = request.get("validation_kind")
        if scene not in {"blocks", "trays"} or kind not in {"x_positive", "y_positive", "angle_zero", "angle_positive_10deg"}:
            raise VisionServiceError("CALIBRATION_VALIDATE_INVALID", "九点方向验证场景或类型无效。")
        path = self._candidates.get((session_id, scene))
        if path is None or not path.is_file():
            raise VisionServiceError("CALIBRATION_CANDIDATE_MISSING", "本次九点候选文件不存在；服务重启后必须重新标定。")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            matrix = np.asarray(value["homography_pixel_to_tool_mm"], dtype=np.float64)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise VisionServiceError("CALIBRATION_CANDIDATE_INVALID", "九点候选文件无法读取。") from exc
        calibration = ApprovedCalibration(
            scene, str(value["calibration_id"]), self.serial, str(value["active_tcp"]),
            str(value["photo_point"]), int(value["image_width"]), int(value["image_height"]), matrix,
            {color: {key: float(item[key]) for key in ("pixel_u", "pixel_v", "r_image_deg", "confidence")}
             for color, item in value.get("reference_detections", {}).items()},
        )
        capture = self.camera.capture(self.profiles[scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "方向验证采集参数未成功写入。")
        target_color = str(value["target_color"])
        captured_at = datetime.now(timezone.utc).isoformat()
        raw_path, image_path, summary = self._save_detection_images(
            session_id, scene, f"validate-{kind}", capture.image_bgr,
            colors=(target_color,), calibration_mode=True, storage_scene=f"calibration/{scene}",
        )
        try:
            detection = locate_calibration_color(capture.image_bgr, profile=self.profiles[scene], calibration=calibration, color=target_color)
        except WorkspaceRecognitionError as exc:
            raise VisionServiceError(
                exc.code, str(exc), details=self._diagnostic_details(scene, raw_path, image_path, summary),
            ) from exc
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "calibration_validate_result", "success": True,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "validation_kind": kind, "camera_serial": self.serial,
            "captured_at": captured_at, "frame_number": capture.frame_number,
            "image_path": str(image_path), "raw_image_path": str(raw_path),
            "annotated_image_path": str(image_path), "detection_summary": summary,
            "detection": detection,
        }

    def _require_calibration(self, session_id: str) -> VisionCalibrationSession:
        if self._calibration is None or self._calibration.session_id != session_id:
            raise VisionServiceError("CALIBRATION_SESSION_MISMATCH", "九点标定会话不存在或编号不匹配。")
        return self._calibration

    def _assert_session_path(self, path: Path) -> None:
        try:
            path.relative_to(self.session_root)
        except ValueError as exc:
            raise VisionServiceError("SESSION_PATH_INVALID", "九点证据路径越界。") from exc

    def _save_detection_images(
        self,
        session_id: str,
        scene: str,
        stem: str,
        image: Any,
        *,
        colors: tuple[str, ...],
        calibration_mode: bool = False,
        selection_policy: str = "strict",
        storage_scene: str | None = None,
    ) -> tuple[Path, Path, dict[str, Any]]:
        directory = (self.session_root / session_id / (storage_scene or scene)).resolve()
        self._assert_session_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / f"{stem}.png"
        annotated_path = directory / f"{stem}-annotated.png"
        if raw_path.exists() or annotated_path.exists():
            raise VisionServiceError("DUPLICATE_CAPTURE", "同一识别步骤的图像已经存在，拒绝覆盖证据。")
        try:
            marked, summary = build_detection_diagnostic(
                image, profile=self.profiles[scene], colors=colors,
                calibration_mode=calibration_mode,
                selection_policy=selection_policy,
            )
        except WorkspaceRecognitionError as exc:
            marked = image.copy()
            cv2.putText(marked, f"DIAGNOSTIC ERROR: {exc.code}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            summary = {
                "success": False,
                "colors": [{"color": color, "status": "configuration_error", "error_code": exc.code, "candidates": [], "selected": None} for color in colors],
            }
        self._write_image_bytes(raw_path, self._encode_png(image, "识别原始图像"), "识别原始图像")
        try:
            self._write_image_bytes(annotated_path, self._encode_png(marked, "识别标注图像"), "识别标注图像")
        except Exception:
            raw_path.unlink(missing_ok=True)
            raise
        return raw_path.resolve(), annotated_path.resolve(), summary

    @staticmethod
    def _diagnostic_details(scene: str, raw_path: Path, annotated_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "scene": scene,
            "image_path": str(raw_path),
            "annotated_image_path": str(annotated_path),
            "detection_summary": summary,
        }

    @staticmethod
    def _encode_png(image: Any, label: str) -> bytes:
        try:
            ok, encoded = cv2.imencode(".png", image)
        except cv2.error as exc:
            raise VisionServiceError("IMAGE_SAVE_FAILED", f"{label}编码失败：{exc}") from exc
        if not ok:
            raise VisionServiceError("IMAGE_SAVE_FAILED", f"{label}编码失败。")
        return encoded.tobytes()

    @staticmethod
    def _write_image_bytes(path: Path, content: bytes, label: str) -> None:
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError as exc:
            raise VisionServiceError("DUPLICATE_CAPTURE", "同一请求的图片已经存在，拒绝覆盖旧证据。") from exc
        except OSError as exc:
            raise VisionServiceError("IMAGE_SAVE_FAILED", f"{label}保存失败：{exc}") from exc

    def _capture_task_card(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"type", "protocol_version", "request_id", "session_id", "scene", "profile", "camera_serial"}
        if set(request) != required or request.get("scene") != "task_card" or request.get("profile") != "task_card":
            raise VisionServiceError("TASK_CAPTURE_INVALID", "任务卡拍照请求字段或 profile不匹配。")
        request_id, session_id = self._ids(request)
        self._assert_serial(request)
        capture = self.camera.capture(self.profiles["task_card"])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "任务卡采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        image_path = self._save_image(session_id, "task_card", request_id, capture.image_bgr)
        return {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "type": "capture_result", "success": True,
            "request_id": request_id, "session_id": session_id,
            "scene": "task_card", "profile": "task_card",
            "data_origin": "camera_vision", "camera_serial": self.serial,
            "captured_at": captured_at, "image_id": f"{request_id}-F{capture.frame_number}",
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "image_path": str(image_path),
            "parameters_applied": True, "configured_parameters": capture.configured_parameters,
        }

    def _capture_and_locate(self, request: dict[str, Any]) -> dict[str, Any]:
        common = {"type", "protocol_version", "request_id", "session_id", "scene", "photo_point", "camera_serial", "calibration_id", "active_tcp"}
        scene = request.get("scene")
        expected_fields = common | ({"target_color"} if scene == "blocks" else set())
        if scene not in {"blocks", "trays"} or set(request) != expected_fields:
            raise VisionServiceError("WORKSPACE_REQUEST_INVALID", "工作区拍照请求字段或场景无效。")
        request_id, session_id = self._ids(request)
        self._assert_serial(request)
        calibration_path = self._single_calibration_file(scene)
        calibration = load_approved_calibration(
            calibration_path, scene=scene, camera_serial=self.serial,
            active_tcp=str(request["active_tcp"]), calibration_id=str(request["calibration_id"]),
        )
        if calibration.photo_point != request.get("photo_point"):
            raise VisionServiceError("PHOTO_POINT_MISMATCH", "真实标定拍照点与本次请求不一致。")
        capture = self.camera.capture(self.profiles[scene])
        if not capture.parameters_applied:
            raise VisionServiceError("PROFILE_APPLY_FAILED", "工作区采集参数未成功写入。")
        captured_at = datetime.now(timezone.utc).isoformat()
        requested_color = request.get("target_color") if scene == "blocks" else None
        colors = (str(requested_color),) if scene == "blocks" else ("红", "橙", "黄", "绿", "蓝", "紫")
        raw_path, annotated_path, summary = self._save_detection_images(
            session_id, scene, request_id, capture.image_bgr, colors=colors,
            selection_policy="best_effort",
        )
        detections: list[dict[str, Any]] = []
        missing_colors: list[str] = []
        for color in colors:
            try:
                detections.extend(locate_colors(
                    capture.image_bgr,
                    profile=self.profiles[scene],
                    calibration=calibration,
                    requested_color=color,
                    selection_policy="best_effort",
                ))
            except WorkspaceRecognitionError as exc:
                if exc.code == "TARGET_NOT_FOUND":
                    missing_colors.append(color)
                    continue
                raise VisionServiceError(
                    exc.code, str(exc),
                    details=self._diagnostic_details(scene, raw_path, annotated_path, summary),
                ) from exc
        if scene == "blocks" and missing_colors:
            raise VisionServiceError(
                "TARGET_NOT_FOUND", f"识别失败：未找到{missing_colors[0]}色目标。",
                details=self._diagnostic_details(scene, raw_path, annotated_path, summary),
            )
        base = {
            "service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id, "scene": scene, "data_origin": "camera_vision",
            "camera_serial": self.serial, "calibration_id": calibration.calibration_id,
            "active_tcp": calibration.active_tcp, "coordinate_frame": "active_tool_at_photo_pose",
            "photo_point": calibration.photo_point,
            "success": True, "captured_at": captured_at,
            "frame_number": capture.frame_number, "image_width": int(capture.image_bgr.shape[1]),
            "image_height": int(capture.image_bgr.shape[0]), "configured_parameters": capture.configured_parameters,
            "image_path": str(raw_path), "annotated_image_path": str(annotated_path),
            "detection_summary": summary,
        }
        if scene == "blocks":
            base.update({"target_color": requested_color, "detection": detections[0]})
        else:
            base["detections"] = detections
            base["missing_colors"] = missing_colors
        return base

    def _ids(self, request: dict[str, Any]) -> tuple[str, str]:
        request_id, session_id = request.get("request_id"), request.get("session_id")
        if not isinstance(request_id, str) or SAFE_ID.fullmatch(request_id) is None or not isinstance(session_id, str) or SAFE_ID.fullmatch(session_id) is None:
            raise VisionServiceError("INVALID_ID", "请求号或 session编号无效。")
        return request_id, session_id

    def _assert_serial(self, request: dict[str, Any]) -> None:
        if request.get("camera_serial") not in {self.serial, self.configured_serial}:
            raise VisionServiceError("CAMERA_IDENTITY_MISMATCH", "请求的相机序列号与当前唯一相机不一致。")

    def _single_calibration_file(self, scene: str) -> Path:
        directory = (self.calibration_root / scene).resolve()
        try:
            directory.relative_to(self.calibration_root)
        except ValueError as exc:
            raise VisionServiceError("CALIBRATION_PATH_INVALID", "真实标定路径越界。") from exc
        files = sorted(directory.glob("*.json"))
        if len(files) != 1:
            raise VisionServiceError("CALIBRATION_NOT_READY", f"识别失败：{scene}必须恰好有一个已批准的真实九点标定文件。")
        return files[0]

    def _save_image(self, session_id: str, scene: str, request_id: str, image: Any) -> Path:
        directory = (self.session_root / session_id / scene).resolve()
        try:
            directory.relative_to(self.session_root)
        except ValueError as exc:
            raise VisionServiceError("SESSION_PATH_INVALID", "图片 session路径越界。") from exc
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{request_id}.png"
        if path.exists():
            raise VisionServiceError("DUPLICATE_CAPTURE", "同一请求号的图片已经存在，拒绝覆盖旧证据。")
        self._write_image_bytes(path, self._encode_png(image, "MVS图像"), "MVS图像")
        return path.resolve()


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(262145)
        if len(line) > 262144:
            response = {"service": SERVICE_NAME, "protocol_version": 1, "type": "error_result", "success": False, "request_id": "", "error_code": "REQUEST_TOO_LARGE", "message": "识别失败：视觉请求超过 256 KiB。"}
        else:
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                request = None
            response = self.server.vision_service.handle(request)  # type: ignore[attr-defined]
        self.wfile.write((json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


class _TcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve() -> int:
    ensure_runtime_directories()
    service = RealMvsVisionService.from_real_config()
    try:
        with _TcpServer((service.endpoint.host, service.endpoint.port), _RequestHandler) as server:
            server.vision_service = service
            print(f"真实 MVS视觉服务已就绪：{service.endpoint.host}:{service.endpoint.port}，相机序列号={service.serial}", flush=True)
            server.serve_forever(poll_interval=0.2)
    finally:
        service.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="装配赛真实 MVS视觉服务")
    parser.add_argument("--check-sdk", action="store_true", help="只检查包内 wrapper和机器级 Runtime，不枚举或打开相机")
    args = parser.parse_args(argv)
    if args.check_sdk:
        print("MVS SDK files: PASS" if sdk_files_available() else "MVS SDK files: FAIL")
        return 0 if sdk_files_available() else 2
    try:
        return serve()
    except (VisionServiceError, MvsCameraError) as exc:
        print(f"真实 MVS视觉服务启动失败：{exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
