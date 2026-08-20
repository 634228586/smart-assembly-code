from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Endpoint
from .protocols import CAPTURE_EVIDENCE_FIELDS, ProtocolError, validate_capture_evidence, validate_service_identity, validate_workspace_result
from vision.contracts import CaptureRequest, CapturedFrame


class VisionClientError(RuntimeError):
    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload


class VisionTargetNotFoundError(VisionClientError):
    """当前新帧没有可返回的目标；装配层只对这一类错误重拍。"""


class RealVisionClient:
    def __init__(self, endpoint: Endpoint, *, active_tcp: str, calibration_ids: dict[str, str] | None = None, timeout_s: float = 5.0, fresh_frame_max_age_ms: int = 5000, visual_result_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.endpoint = endpoint
        self.active_tcp = active_tcp
        self.calibration_ids = calibration_ids or {}
        self.timeout_s = timeout_s
        self.fresh_frame_max_age_ms = int(fresh_frame_max_age_ms)
        self.visual_result_callback = visual_result_callback or (lambda _payload: None)

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with socket.create_connection((self.endpoint.host, self.endpoint.port), timeout=self.timeout_s) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall((json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                data = bytearray()
                while not data.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        raise VisionClientError("视觉服务提前关闭连接。")
                    data.extend(chunk)
                    if len(data) > 262144:
                        raise VisionClientError("视觉响应超过 256 KiB。")
            result = json.loads(data.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisionClientError(f"视觉通信失败：{exc}") from exc
        if not isinstance(result, dict):
            raise VisionClientError("视觉响应不是 JSON对象。")
        if isinstance(result.get("annotated_image_path"), str):
            self.visual_result_callback(result)
        if result.get("success") is False:
            message = result.get("message")
            code = result.get("error_code")
            if not isinstance(message, str) or not message.strip() or not isinstance(code, str):
                raise VisionClientError("识别失败：视觉服务返回了无效失败响应。")
            prefix = "" if message.startswith("识别失败") else "识别失败："
            error_type = VisionTargetNotFoundError if code == "TARGET_NOT_FOUND" else VisionClientError
            raise error_type(f"{prefix}{message}（{code}）", payload=result)
        return result

    def health(self) -> dict[str, Any]:
        payload = self._exchange({"type": "health_request", "protocol_version": 1, "expected_service": self.endpoint.expected_service})
        try:
            validate_service_identity(payload, expected_service=self.endpoint.expected_service)
        except ProtocolError as exc:
            raise VisionClientError(str(exc)) from exc
        if payload.get("source") != "mvs" or payload.get("mounting") != "eye_in_hand":
            raise VisionClientError("视觉服务来源或安装方式不匹配。")
        return payload

    def capture_task_card(self, *, request_id: str, session_id: str, session_dir: Path) -> CapturedFrame:
        """要求统一 MVS 服务切换 task_card profile 并保存本次新图。"""

        request = CaptureRequest(request_id, session_dir.resolve(), "task_card")
        request.validate()
        started_at = datetime.now(timezone.utc)
        result = self._exchange({
            "type": "capture_frame", "protocol_version": 1, "request_id": request_id,
            "session_id": session_id, "scene": "task_card", "profile": "task_card",
        })
        required = {
            "service", "protocol_version", "type", "success", "request_id", "session_id",
            "scene", "profile", "data_origin", "captured_at", "image_id",
            "image_path", "parameters_applied",
        } | CAPTURE_EVIDENCE_FIELDS
        if not isinstance(result, dict) or set(result) != required:
            raise VisionClientError("任务卡拍照响应字段不完整或包含未知字段。")
        expected = {
            "service": self.endpoint.expected_service, "protocol_version": 1,
            "type": "capture_result", "success": True, "request_id": request_id,
            "session_id": session_id, "scene": "task_card", "profile": "task_card",
            "data_origin": "camera_vision",
            "parameters_applied": True,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise VisionClientError("任务卡拍照响应身份、来源或采集参数不匹配。")
        try:
            validate_capture_evidence(result)
        except ProtocolError as exc:
            raise VisionClientError(str(exc)) from exc
        try:
            captured_at = datetime.fromisoformat(str(result["captured_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise VisionClientError("任务卡拍摄时间无效。") from exc
        if captured_at.tzinfo is None or captured_at.astimezone(timezone.utc) < started_at:
            raise VisionClientError("任务卡服务返回了本次请求之前的旧帧。")
        age_ms = (datetime.now(timezone.utc) - captured_at.astimezone(timezone.utc)).total_seconds() * 1000
        if age_ms > self.fresh_frame_max_age_ms or age_ms < -5000:
            raise VisionClientError("任务卡帧不满足新鲜度要求。")
        frame = CapturedFrame(
            request_id=request_id, profile="task_card",
            image_path=Path(str(result["image_path"])).resolve(), captured_at=str(result["captured_at"]),
            parameters_applied=True,
        )
        frame.validate_for(request)
        return frame

    def locate_block(self, *, request_id: str, color: str, photo_point: str, session_id: str) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        result = self._exchange({
            "type": "capture_and_locate", "protocol_version": 1, "request_id": request_id,
            "session_id": session_id, "scene": "blocks", "target_color": color,
            "photo_point": photo_point,
            "calibration_id": self.calibration_ids["blocks"], "active_tcp": self.active_tcp,
        })
        return validate_workspace_result(result, scene="blocks", request_id=request_id, calibration_id=self.calibration_ids["blocks"], active_tcp=self.active_tcp, photo_point=photo_point, request_started_at=started_at, fresh_frame_max_age_ms=self.fresh_frame_max_age_ms)["detection"]

    def locate_trays(self, *, request_id: str, photo_point: str, session_id: str) -> dict[str, dict[str, Any]]:
        started_at = datetime.now(timezone.utc)
        result = self._exchange({
            "type": "capture_and_locate", "protocol_version": 1, "request_id": request_id,
            "session_id": session_id, "scene": "trays", "photo_point": photo_point,
            "calibration_id": self.calibration_ids["trays"],
            "active_tcp": self.active_tcp,
        })
        validated = validate_workspace_result(result, scene="trays", request_id=request_id, calibration_id=self.calibration_ids["trays"], active_tcp=self.active_tcp, photo_point=photo_point, request_started_at=started_at, fresh_frame_max_age_ms=self.fresh_frame_max_age_ms)
        return {item["color"]: item for item in validated["detections"]}

    def calibration_begin(self, *, request_id: str, session_id: str, scene: str, target_color: str, photo_point: str, robot_serial: str, step_x_mm: float, step_y_mm: float) -> dict[str, Any]:
        result = self._exchange({
            "type": "calibration_begin", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "scene": scene, "profile": scene, "target_color": target_color,
            "photo_point": photo_point,
            "robot_serial": robot_serial, "active_tcp": self.active_tcp,
            "step_x_mm": float(step_x_mm), "step_y_mm": float(step_y_mm),
        })
        self._assert_calibration_identity(result, "calibration_begin_result", request_id, session_id)
        if result.get("scene") != scene:
            raise VisionClientError("九点开始响应的场景不匹配。")
        return result

    def validate_detector(self, *, request_id: str, session_id: str, scene: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "detector_validate", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "scene": scene,
        })
        self._assert_calibration_identity(result, "detector_validate_result", request_id, session_id)
        if result.get("scene") != scene:
            raise VisionClientError("颜色参数验证响应身份不匹配。")
        detections = result.get("detections")
        if not isinstance(detections, list) or {item.get("color") for item in detections if isinstance(item, dict)} != {"红", "橙", "黄", "绿", "蓝", "紫"}:
            raise VisionClientError("颜色参数验证未唯一覆盖六色。")
        return result

    def estimate_detector_area(self, *, request_id: str, session_id: str, scene: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "detector_estimate_area", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "scene": scene,
        })
        self._assert_calibration_identity(result, "detector_estimate_area_result", request_id, session_id)
        if result.get("scene") != scene:
            raise VisionClientError("面积自动估算响应身份不匹配。")
        areas = result.get("areas_px")
        if not isinstance(areas, dict) or set(areas) != {"红", "橙", "黄", "绿", "蓝", "紫"}:
            raise VisionClientError("面积自动估算未唯一覆盖六色。")
        try:
            minimum, maximum = float(result["min_area_px"]), float(result["max_area_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VisionClientError("面积自动估算范围无效。") from exc
        if not 0 < minimum < maximum:
            raise VisionClientError("面积自动估算范围无效。")
        return result

    def capture_manual_scene(self, *, request_id: str, session_id: str, scene: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "manual_scene_capture", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "scene": scene,
        })
        self._assert_calibration_identity(result, "manual_scene_capture_result", request_id, session_id)
        if scene not in {"blocks", "trays"} or result.get("scene") != scene:
            raise VisionClientError("手动取图响应场景不匹配。")
        if result.get("parameters_applied") is not True:
            raise VisionClientError("手动取图响应的采集参数状态不匹配。")
        image_path = result.get("image_path")
        if not isinstance(image_path, str) or not Path(image_path).resolve().is_file():
            raise VisionClientError("手动取图响应缺少有效的原图路径。")
        return result

    def recognize_manual_scene(self, *, request_id: str, session_id: str, scene: str) -> dict[str, Any]:
        """Capture one new frame and diagnose all colors without approving configuration."""

        if scene not in {"blocks", "trays"}:
            raise VisionClientError("手动拍照识别场景无效。")
        result = self._exchange({
            "type": "manual_scene_recognize", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id, "scene": scene,
            "active_tcp": self.active_tcp,
        })
        required = {
            "type": "manual_scene_recognize_result", "request_id": request_id,
            "session_id": session_id, "scene": scene,
            "active_tcp": self.active_tcp, "success": True,
        }
        if any(result.get(key) != value for key, value in required.items()):
            raise VisionClientError("手动拍照识别响应身份或类型无效。")
        for field in ("image_path", "annotated_image_path"):
            value = result.get(field)
            if not isinstance(value, str) or not Path(value).resolve().is_file():
                raise VisionClientError(f"手动拍照识别响应缺少有效{field}。")
        detections = result.get("detections")
        missing = result.get("missing_colors")
        if not isinstance(detections, list) or not all(isinstance(item, dict) for item in detections):
            raise VisionClientError("手动拍照识别 detections无效。")
        valid_colors = {"红", "橙", "黄", "绿", "蓝", "紫"}
        if not isinstance(missing, list) or any(color not in valid_colors for color in missing):
            raise VisionClientError("手动拍照识别 missing_colors无效。")
        if {item.get("color") for item in detections} & set(missing):
            raise VisionClientError("手动拍照识别的成功颜色与缺失颜色冲突。")
        return result

    def validate_profile(self, *, request_id: str, session_id: str, profile: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "profile_validate", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "profile": profile,
        })
        self._assert_calibration_identity(result, "profile_validate_result", request_id, session_id)
        if result.get("profile") != profile or result.get("parameters_applied") is not True:
            raise VisionClientError("采集参数写入测试响应身份或状态不匹配。")
        if not isinstance(result.get("configured_parameters"), dict) or not isinstance(result.get("profile_sha256"), str):
            raise VisionClientError("采集参数写入测试响应缺少配置值或配置哈希。")
        return result

    def calibration_capture_point(self, *, request_id: str, session_id: str, index: int, actual_tcp_pose: list[float], tool_x_mm: float, tool_y_mm: float) -> dict[str, Any]:
        result = self._exchange({
            "type": "calibration_capture_point", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "index": int(index),
            "actual_tcp_pose": [float(value) for value in actual_tcp_pose],
            "tool_x_mm": float(tool_x_mm), "tool_y_mm": float(tool_y_mm),
        })
        self._assert_calibration_identity(result, "calibration_point_result", request_id, session_id)
        if result.get("accepted_count") != index:
            raise VisionClientError("九点采集响应累计数量不匹配。")
        return result

    def calibration_finish(self, *, request_id: str, session_id: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "calibration_finish", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
        })
        self._assert_calibration_identity(result, "calibration_finish_result", request_id, session_id)
        if result.get("approved") is not False:
            raise VisionClientError("九点服务不得把未验证候选标为已批准。")
        return result

    def calibration_validate_capture(self, *, request_id: str, session_id: str, scene: str, validation_kind: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "calibration_validate_capture", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
            "scene": scene, "validation_kind": validation_kind,
        })
        self._assert_calibration_identity(result, "calibration_validate_result", request_id, session_id)
        if result.get("scene") != scene or result.get("validation_kind") != validation_kind:
            raise VisionClientError("九点方向验证响应场景或类型不匹配。")
        return result

    def calibration_abort(self, *, request_id: str, session_id: str) -> dict[str, Any]:
        result = self._exchange({
            "type": "calibration_abort", "protocol_version": 1,
            "request_id": request_id, "session_id": session_id,
        })
        self._assert_calibration_identity(result, "calibration_abort_result", request_id, session_id)
        return result

    def _assert_calibration_identity(self, result: dict[str, Any], response_type: str, request_id: str, session_id: str) -> None:
        expected = {
            "service": self.endpoint.expected_service, "protocol_version": 1,
            "type": response_type, "success": True,
            "request_id": request_id, "session_id": session_id,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise VisionClientError("九点视觉响应身份、类型或请求号不匹配。")
