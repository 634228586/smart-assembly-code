from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLORS = {"红", "橙", "黄", "绿", "蓝", "紫"}
REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


class ProtocolError(ValueError):
    pass


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolError(f"{field} 必须是 ISO 时间字符串。")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(f"{field} 不是有效 ISO 时间。") from exc
    if parsed.tzinfo is None:
        raise ProtocolError(f"{field} 必须带时区。")
    return parsed


def validate_recognition_result(payload: Any, *, session_dir: Path, camera_serial: str, now: datetime | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("识别结果必须是 JSON 对象。")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
        raise ProtocolError("request_id 无效。")
    if payload.get("schema_version") != 1 or payload.get("type") != "recognition_result":
        raise ProtocolError("识别协议版本或类型无效。")
    task_type = payload.get("task_type")
    if task_type not in {"task_1", "task_2", "unknown"} or not isinstance(payload.get("success"), bool):
        raise ProtocolError("task_type 或 success 无效。")
    if payload["success"] is True and task_type == "unknown":
        raise ProtocolError("识别成功时 task_type 不能为 unknown。")
    source = payload.get("source_image")
    required_source = {"image_id", "path", "captured_at", "camera_serial", "capture_request_id"}
    if not isinstance(source, dict) or set(source) != required_source:
        raise ProtocolError("source_image 必须包含真实采集身份字段。")
    image_path = Path(str(source.get("path", ""))).resolve()
    try:
        image_path.relative_to(session_dir.resolve())
    except ValueError as exc:
        raise ProtocolError("任务卡图片不属于本场 session。") from exc
    if not image_path.is_file():
        raise ProtocolError("本次任务卡图片文件不存在。")
    if source.get("camera_serial") != camera_serial or source.get("capture_request_id") != request_id:
        raise ProtocolError("任务卡相机或拍照请求号不匹配。")
    captured = _timestamp(source.get("captured_at"), "source_image.captured_at")
    recognized = _timestamp(payload.get("recognized_at"), "recognized_at")
    if recognized < captured:
        raise ProtocolError("识别时间早于拍摄时间。")
    current = now or datetime.now(timezone.utc)
    age_s = (current - captured.astimezone(timezone.utc)).total_seconds()
    if age_s < -5:
        raise ProtocolError("任务卡图片拍摄时间晚于当前时间。")
    if age_s > 120:
        raise ProtocolError("任务卡图片不是本次新鲜拍摄。")
    if payload["success"] is False:
        expected_fields = {"schema_version", "type", "request_id", "task_type", "success", "recognized_at", "raw_text", "source_image", "error_code", "message"}
        if set(payload) != expected_fields:
            raise ProtocolError("失败结果字段不完整或包含未知字段。")
        if not isinstance(payload.get("error_code"), str) or not isinstance(payload.get("message"), str):
            raise ProtocolError("失败结果必须提供 error_code 和 message。")
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ProtocolError("confidence 无效。")
    if task_type == "task_1":
        expected_fields = {"schema_version", "type", "request_id", "task_type", "success", "confidence", "recognized_at", "raw_text", "source_image", "scene_description"}
        if set(payload) != expected_fields:
            raise ProtocolError("任务一卡字段不完整或包含未知字段。")
        if not isinstance(payload.get("scene_description"), str) or not payload["scene_description"].strip() or "sequence" in payload:
            raise ProtocolError("任务一卡缺少场景描述。")
    else:
        expected_fields = {"schema_version", "type", "request_id", "task_type", "success", "confidence", "recognized_at", "raw_text", "source_image", "sequence"}
        if set(payload) != expected_fields:
            raise ProtocolError("任务二卡字段不完整或包含未知字段。")
        sequence = payload.get("sequence")
        if not isinstance(sequence, list) or len(sequence) != 6:
            raise ProtocolError("任务二必须恰好包含六组。")
        orders, blocks, trays = set(), set(), set()
        for step in sequence:
            if not isinstance(step, dict) or set(step) != {"order", "block_color", "tray_color"}:
                raise ProtocolError("任务二步骤字段无效。")
            orders.add(step["order"]); blocks.add(step["block_color"]); trays.add(step["tray_color"])
        if orders != set(range(1, 7)) or blocks != COLORS or trays != COLORS:
            raise ProtocolError("任务二必须完整且唯一覆盖六个颜色和顺序 1..6。")
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def validate_service_identity(payload: Any, *, expected_service: str, protocol_version: int = 1) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("服务健康响应必须是 JSON 对象。")
    if payload.get("service") != expected_service or payload.get("protocol_version") != protocol_version:
        raise ProtocolError("服务身份或协议版本不匹配。")
    if payload.get("status") != "ready":
        raise ProtocolError("服务尚未 ready。")
    return payload


def normalize_speech_health(payload: Any, *, expected_service: str) -> dict[str, Any]:
    """Strictly adapt the verified m28 real-box health schema to protocol v1."""

    if not isinstance(payload, dict) or payload.get("service") != expected_service:
        raise ProtocolError("语音服务身份不匹配。")
    if expected_service != "arm_speech_service":
        validated = validate_service_identity(payload, expected_service=expected_service)
        capabilities = validated.get("capabilities")
        if not isinstance(capabilities, list) or not {"asr", "tts"}.issubset(capabilities):
            raise ProtocolError("语音服务未声明 ASR/TTS能力。")
        return validated

    if payload.get("version") != "m28-4-real" or payload.get("provider") != "real" or payload.get("ready") is not True:
        raise ProtocolError("AI语音盒子版本、真实来源或 ready状态不匹配。")
    if payload.get("capture_ready") not in {True, "ready", "lazy"}:
        raise ProtocolError("AI语音盒子采集设备未就绪。")
    models = payload.get("models")
    if not isinstance(models, dict) or any(models.get(name) is not True for name in ("asr_exists", "keyword_exists", "vad_exists")):
        raise ProtocolError("AI语音盒子的 ASR、唤醒词或 VAD模型未就绪。")
    backends = payload.get("tts_backends")
    if not isinstance(backends, list) or not any(isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"] for item in backends):
        raise ProtocolError("AI语音盒子没有可用 TTS后端。")
    if models.get("piper_exists") is not True:
        raise ProtocolError("AI语音盒子的离线 Piper TTS模型未就绪。")

    normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    normalized.update({"protocol_version": 1, "status": "ready", "capabilities": ["asr", "tts"]})
    normalized["compatibility_adapter"] = "arm_speech_m28_v1"
    return normalized


CAPTURE_EVIDENCE_FIELDS = {"frame_number", "image_width", "image_height", "configured_parameters"}
WORKSPACE_VISUAL_FIELDS = {"image_path", "annotated_image_path", "detection_summary"}


def validate_capture_evidence(payload: dict[str, Any]) -> None:
    """Validate evidence that the response came from the newly triggered MVS frame."""

    for field, minimum in (("frame_number", 0), ("image_width", 1), ("image_height", 1)):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ProtocolError(f"视觉新帧证据 {field} 无效。")
    configured = payload.get("configured_parameters")
    if not isinstance(configured, dict) or not configured:
        raise ProtocolError("视觉新帧缺少本次写入的相机参数。")
    roi = configured.get("roi")
    if not isinstance(roi, dict):
        raise ProtocolError("视觉新帧的配置 ROI 无效。")
    if roi.get("width") != payload["image_width"] or roi.get("height") != payload["image_height"]:
        raise ProtocolError("视觉新帧尺寸与本次写入的 ROI 不一致。")


def validate_workspace_result(
    payload: Any,
    *,
    scene: str,
    request_id: str,
    camera_serial: str,
    calibration_id: str,
    active_tcp: str,
    photo_point: str,
    request_started_at: datetime | None = None,
    now: datetime | None = None,
    fresh_frame_max_age_ms: int = 5000,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("视觉结果必须是 JSON 对象。")
    required = {
        "service": "real_mvs_vision", "protocol_version": 1, "request_id": request_id,
        "scene": scene, "data_origin": "camera_vision", "camera_serial": camera_serial,
        "calibration_id": calibration_id, "active_tcp": active_tcp,
        "photo_point": photo_point,
        "coordinate_frame": "active_tool_at_photo_pose", "success": True,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ProtocolError(f"视觉字段 {key} 与本次真实请求不匹配。")
    validate_capture_evidence(payload)
    for field in ("image_path", "annotated_image_path"):
        path = Path(str(payload.get(field, "")))
        if not path.is_file():
            raise ProtocolError(f"视觉结果图像 {field} 不存在。")
    summary = payload.get("detection_summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("colors"), list):
        raise ProtocolError("视觉结果缺少带框检测摘要。")
    if scene == "blocks" and summary.get("success") is not True:
        raise ProtocolError("方块视觉结果缺少成功的带框检测摘要。")
    captured = _timestamp(payload.get("captured_at"), "captured_at").astimezone(timezone.utc)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if captured > current.replace(microsecond=current.microsecond) and (captured - current).total_seconds() > 5:
        raise ProtocolError("视觉帧拍摄时间晚于当前时间。")
    if request_started_at is not None and captured < request_started_at.astimezone(timezone.utc):
        raise ProtocolError("视觉帧早于本次拍照请求，拒绝旧帧。")
    # Encoding two full-resolution PNG files and running color detection is
    # post-capture work. Allow that bounded processing time, while the request
    # timestamp check above still guarantees this request cannot reuse an old frame.
    if (current - captured).total_seconds() * 1000 > fresh_frame_max_age_ms:
        raise ProtocolError("视觉帧超过允许的新鲜度。")
    def detection(value: Any) -> dict[str, Any]:
        expected = {
            "color", "dx_tool_m", "dy_tool_m", "r_image_rad", "confidence",
            "delta_x_tool_m", "delta_y_tool_m", "delta_r_rad",
            "reference_pixel_u", "reference_pixel_v", "current_pixel_u", "current_pixel_v",
        }
        if not isinstance(value, dict) or set(value) != expected or value.get("color") not in COLORS:
            raise ProtocolError("视觉 detection字段或颜色无效。")
        for field in expected - {"color"}:
            number = value[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ProtocolError(f"视觉 detection.{field}不是有限数值。")
        if not -math.pi / 4 <= value["r_image_rad"] < math.pi / 4 or not -math.pi / 4 <= value["delta_r_rad"] < math.pi / 4 or not 0 <= value["confidence"] <= 1:
            raise ProtocolError("视觉角度或置信度超出范围。")
        return value
    if scene == "blocks":
        if set(payload) != set(required) | CAPTURE_EVIDENCE_FIELDS | WORKSPACE_VISUAL_FIELDS | {"captured_at", "target_color", "detection"}:
            raise ProtocolError("方块视觉结果字段不匹配。")
        item = detection(payload.get("detection"))
        if payload.get("target_color") != item["color"]:
            raise ProtocolError("方块目标颜色与 detection不一致。")
    elif scene == "trays":
        if set(payload) != set(required) | CAPTURE_EVIDENCE_FIELDS | WORKSPACE_VISUAL_FIELDS | {"captured_at", "detections", "missing_colors"}:
            raise ProtocolError("托盘视觉结果字段不匹配。")
        items = payload.get("detections")
        missing = payload.get("missing_colors")
        if not isinstance(items, list) or not isinstance(missing, list):
            raise ProtocolError("托盘视觉结果或缺失颜色列表无效。")
        colors = [detection(item)["color"] for item in items]
        if any(color not in COLORS for color in missing):
            raise ProtocolError("托盘缺失颜色列表含有未知颜色。")
        if len(set(colors)) != len(colors) or len(set(missing)) != len(missing):
            raise ProtocolError("托盘视觉结果或缺失颜色存在重复。")
        if set(colors) & set(missing) or set(colors) | set(missing) != COLORS:
            raise ProtocolError("托盘已识别颜色与缺失颜色未唯一覆盖六色。")
    else:
        raise ProtocolError("未知工作区场景。")
    return payload
