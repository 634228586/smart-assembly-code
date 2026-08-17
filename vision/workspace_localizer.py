from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = ("红", "橙", "黄", "绿", "蓝", "紫")
COLOR_LABELS = {"红": "RED", "橙": "ORANGE", "黄": "YELLOW", "绿": "GREEN", "蓝": "BLUE", "紫": "PURPLE"}


class WorkspaceRecognitionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ApprovedCalibration:
    scene: str
    calibration_id: str
    camera_serial: str
    active_tcp: str
    photo_point: str
    image_width: int
    image_height: int
    homography: np.ndarray
    reference_detections: dict[str, dict[str, float]]


def load_approved_calibration(path: Path, *, scene: str, camera_serial: str, active_tcp: str, calibration_id: str) -> ApprovedCalibration:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceRecognitionError("CALIBRATION_READ_FAILED", f"识别失败：真实标定文件无法读取：{exc}") from exc
    expected = {
        "schema_version": 1, "scene": scene, "data_origin": "camera_vision",
        "usable_for_real_robot": True, "approved": True,
        "camera_serial": camera_serial, "active_tcp": active_tcp,
        "calibration_id": calibration_id,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise WorkspaceRecognitionError("CALIBRATION_IDENTITY_MISMATCH", "识别失败：真实标定身份与本次请求不一致。")
    try:
        width, height = int(raw["image_width"]), int(raw["image_height"])
        photo_point = str(raw["photo_point"])
        matrix = np.asarray(raw["homography_pixel_to_tool_mm"], dtype=np.float64)
        reference_raw = raw["reference_detections"]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：真实标定缺少分辨率、拍照点或九点矩阵。") from exc
    if width <= 0 or height <= 0 or not photo_point or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：真实标定矩阵或图像尺寸无效。")
    if not isinstance(reference_raw, dict) or set(reference_raw) != set(COLORS):
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：真实标定缺少六色基准中心和角度。")
    references: dict[str, dict[str, float]] = {}
    try:
        for color in COLORS:
            item = reference_raw[color]
            references[color] = {key: float(item[key]) for key in ("pixel_u", "pixel_v", "r_image_deg", "confidence")}
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：六色基准中心或角度字段无效。") from exc
    if not all(np.isfinite(list(item.values())).all() for item in references.values()):
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：六色基准中心或角度包含非有限数值。")
    return ApprovedCalibration(scene, calibration_id, camera_serial, active_tcp, photo_point, width, height, matrix, references)


def locate_colors(
    image_bgr: np.ndarray,
    *,
    profile: dict[str, Any],
    calibration: ApprovedCalibration,
    requested_color: str | None = None,
    selection_policy: str = "strict",
) -> list[dict[str, Any]]:
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise WorkspaceRecognitionError("INVALID_FRAME", "识别失败：相机图像不是有效 BGR彩色帧。")
    if (image.shape[1], image.shape[0]) != (calibration.image_width, calibration.image_height):
        raise WorkspaceRecognitionError("RESOLUTION_MISMATCH", "识别失败：相机分辨率与真实九点标定不一致。")
    detector = profile.get("detector")
    if not isinstance(detector, dict):
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：本场景颜色识别参数缺失。")
    colors = (requested_color,) if requested_color is not None else COLORS
    if any(color not in COLORS for color in colors):
        raise WorkspaceRecognitionError("UNKNOWN_COLOR", "识别失败：请求了未知颜色。")
    return [
        _detect_one(
            image, color, detector, calibration.homography,
            reference_detections=calibration.reference_detections,
            selection_policy=selection_policy,
        )
        for color in colors
    ]


def detect_color_pixel(
    image_bgr: np.ndarray,
    *,
    profile: dict[str, Any],
    color: str,
    require_approved: bool = True,
    selection_policy: str = "strict",
) -> dict[str, Any]:
    """Detect one calibration target without requiring an existing homography."""

    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise WorkspaceRecognitionError("INVALID_FRAME", "识别失败：相机图像不是有效 BGR彩色帧。")
    if color not in COLORS:
        raise WorkspaceRecognitionError("UNKNOWN_COLOR", "识别失败：九点目标颜色无效。")
    detector = profile.get("detector")
    if not isinstance(detector, dict) or (require_approved and detector.get("approved") is not True):
        raise WorkspaceRecognitionError("DETECTOR_NOT_APPROVED", "识别失败：本场景颜色识别参数尚未批准。")
    confidence, center, image_angle = _pixel_candidate(
        image, color, detector, selection_policy=selection_policy,
    )
    return {
        "color": color,
        "pixel_u": center[0],
        "pixel_v": center[1],
        "r_image_deg": image_angle,
        "confidence": confidence,
    }


def _single_color_calibration_detector(image: np.ndarray, profile: dict[str, Any]) -> dict[str, Any]:
    """九点专用单色检测器：不依赖六色批准或既有面积阈值，仍强制唯一方形候选。"""

    detector = profile.get("detector")
    if not isinstance(detector, dict) or not isinstance(detector.get("hsv_ranges"), dict):
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：九点单色HSV配置缺失。")
    height, width = image.shape[:2]
    frame_area = float(width * height)
    value = dict(detector)
    value.update({
        "roi": [0, 0, width, height],
        "min_area_px": max(25.0, frame_area * 0.00001),
        "max_area_px": frame_area * 0.25,
        "confidence_min": float(detector.get("confidence_min", 0.6)),
    })
    return value


def detect_calibration_color_pixel(image_bgr: np.ndarray, *, profile: dict[str, Any], color: str) -> dict[str, Any]:
    """只检测九点页选定的单色目标，不要求 blocks/trays 六色批准。"""

    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise WorkspaceRecognitionError("INVALID_FRAME", "识别失败：相机图像不是有效 BGR彩色帧。")
    if color not in COLORS:
        raise WorkspaceRecognitionError("UNKNOWN_COLOR", "识别失败：九点目标颜色无效。")
    confidence, center, image_angle = _pixel_candidate(image, color, _single_color_calibration_detector(image, profile))
    return {
        "color": color, "pixel_u": center[0], "pixel_v": center[1],
        "r_image_deg": image_angle, "confidence": confidence,
    }


def locate_calibration_color(
    image_bgr: np.ndarray, *, profile: dict[str, Any], calibration: ApprovedCalibration, color: str
) -> dict[str, Any]:
    """使用候选九点矩阵换算单色验证目标，不要求六色批准。"""

    image = np.asarray(image_bgr)
    if (image.shape[1], image.shape[0]) != (calibration.image_width, calibration.image_height):
        raise WorkspaceRecognitionError("RESOLUTION_MISMATCH", "识别失败：相机分辨率与九点候选不一致。")
    return _detect_one(image, color, _single_color_calibration_detector(image, profile), calibration.homography)


def _detect_one(
    image: np.ndarray,
    color: str,
    config: dict[str, Any],
    homography: np.ndarray,
    *,
    selection_policy: str = "strict",
    reference_detections: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    confidence, center, image_angle = _pixel_candidate(
        image, color, config, selection_policy=selection_policy,
    )
    mapped, angle_rad = _map_center_and_angle(center, image_angle, homography)
    if not np.isfinite(mapped).all():
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：九点映射产生非有限数值。")
    current_xy = mapped[0]
    if reference_detections is None:
        anchor_xy = np.asarray([0.0, 0.0], dtype=np.float64)
        color_xy = anchor_xy
        reference_angle = 0.0
        reference = {"pixel_u": center[0], "pixel_v": center[1]}
    else:
        reference = reference_detections[color]
        anchor = reference_detections["红"]
        anchor_mapped, _ = _map_center_and_angle((anchor["pixel_u"], anchor["pixel_v"]), anchor["r_image_deg"], homography)
        color_mapped, reference_angle = _map_center_and_angle((reference["pixel_u"], reference["pixel_v"]), reference["r_image_deg"], homography)
        anchor_xy = anchor_mapped[0]
        color_xy = color_mapped[0]
    anchor_delta = current_xy - anchor_xy
    color_delta = current_xy - color_xy
    return {
        "color": color,
        "dx_tool_m": float(anchor_delta[0]) / 1000.0,
        "dy_tool_m": float(anchor_delta[1]) / 1000.0,
        "r_image_rad": angle_rad,
        "confidence": confidence,
        "delta_x_tool_m": float(color_delta[0]) / 1000.0,
        "delta_y_tool_m": float(color_delta[1]) / 1000.0,
        "delta_r_rad": math.radians(_normalize_square(math.degrees(angle_rad - reference_angle))),
        "reference_pixel_u": float(reference["pixel_u"]),
        "reference_pixel_v": float(reference["pixel_v"]),
        "current_pixel_u": float(center[0]),
        "current_pixel_v": float(center[1]),
    }


def _map_center_and_angle(center: tuple[float, float], image_angle_deg: float, homography: np.ndarray) -> tuple[np.ndarray, float]:
    theta = math.radians(image_angle_deg)
    points = np.asarray([[center, (center[0] + 20 * math.cos(theta), center[1] + 20 * math.sin(theta))]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(points, homography)[0]
    if not np.isfinite(mapped).all():
        raise WorkspaceRecognitionError("CALIBRATION_INVALID", "识别失败：九点映射产生非有限数值。")
    delta = mapped[1] - mapped[0]
    if float(np.linalg.norm(delta)) <= 1e-9:
        raise WorkspaceRecognitionError("ANGLE_UNAVAILABLE", "识别失败：九点映射无法确定目标方向。")
    angle_rad = math.radians(_normalize_square(math.degrees(math.atan2(float(delta[1]), float(delta[0])))))
    return mapped, angle_rad


def analyze_color_candidates(
    image: np.ndarray,
    color: str,
    config: dict[str, Any],
    *,
    selection_policy: str = "strict",
) -> dict[str, Any]:
    """Return visible contours and the target selected by the requested policy.

    ``strict`` preserves the calibration/formal legacy gates. ``best_effort`` is
    the offline competition-preview policy: every HSV contour above the tiny
    fragment floor is eligible, area is the primary sort key, and confidence
    breaks area ties. The legacy area/shape/confidence checks remain attached
    as warnings.
    """

    if selection_policy not in {"strict", "best_effort"}:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：未知的候选选择规则。")

    try:
        roi = tuple(int(value) for value in config["roi"])
        ranges = config["hsv_ranges"][color]
        min_area = float(config["min_area_px"]); max_area = float(config["max_area_px"])
        confidence_min = float(config["confidence_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：颜色识别参数字段无效。") from exc
    if len(roi) != 4:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：颜色识别 ROI必须包含四个整数。")
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 0 < min_area < max_area and 0 <= confidence_min <= 1):
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：颜色识别 ROI、面积或置信度参数无效。")
    crop = cv2.GaussianBlur(image[y1:y2, x1:x2], (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    if not isinstance(ranges, list) or not ranges:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色 HSV范围缺失。")
    for band in ranges:
        try:
            lower = np.asarray(band["lower"], dtype=np.uint8); upper = np.asarray(band["upper"], dtype=np.uint8)
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色 HSV范围无效。") from exc
        if lower.shape != (3,) or upper.shape != (3,):
            raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色 HSV范围必须是三通道。")
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observations: list[dict[str, Any]] = []
    for local in contours:
        contour = np.array(local, dtype=np.int32, copy=True)
        contour[:, 0, 0] += x1; contour[:, 0, 1] += y1
        area = float(cv2.contourArea(contour))
        (cx, cy), (rect_w, rect_h), raw_angle = cv2.minAreaRect(contour)
        aspect = 0.0 if max(rect_w, rect_h) <= 0 else min(rect_w, rect_h) / max(rect_w, rect_h)
        rectangularity = 0.0 if rect_w * rect_h <= 0 else min(1.0, max(0.0, area / (rect_w * rect_h)))
        confidence = float(0.65 * aspect + 0.35 * rectangularity)
        reasons: list[str] = []
        if area < min_area: reasons.append("area_below_min")
        if area > max_area: reasons.append("area_above_max")
        if min(rect_w, rect_h) < 3 or max(rect_w, rect_h) <= 0: reasons.append("too_small")
        if aspect < 0.65: reasons.append("not_square")
        if confidence < confidence_min: reasons.append("confidence_below_min")
        box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), raw_angle))
        observations.append({
            "center": [float(cx), float(cy)],
            "box": [[int(round(point[0])), int(round(point[1]))] for point in box],
            "area_px": area,
            "aspect": float(aspect),
            "rectangularity": float(rectangularity),
            "confidence": confidence,
            "angle_deg": _normalize_square(float(raw_angle)),
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "selected": False,
        })
    fragment_area_floor = max(1.0, min_area * 0.05)
    eligible: list[dict[str, Any]] = []
    if selection_policy == "best_effort":
        for item in observations:
            item["ignored_as_tiny_fragment"] = float(item["area_px"]) < fragment_area_floor
        eligible = [item for item in observations if not item["ignored_as_tiny_fragment"]]
        eligible.sort(key=lambda item: (float(item["area_px"]), float(item["confidence"])), reverse=True)
        ignored = [item for item in observations if item["ignored_as_tiny_fragment"]]
        observations = eligible + sorted(
            ignored,
            key=lambda item: (float(item["confidence"]), float(item["area_px"])),
            reverse=True,
        )
    else:
        for item in observations:
            item["ignored_as_tiny_fragment"] = False
        observations.sort(key=lambda item: (bool(item["accepted"]), float(item["confidence"]), float(item["area_px"])), reverse=True)
    accepted = [item for item in observations if item["accepted"]]
    status, error_code = "success", None
    warnings: list[str] = []
    if selection_policy == "best_effort":
        ignored_count = len(observations) - len(eligible)
        if ignored_count:
            warnings.append("tiny_fragments_ignored")
        if not eligible:
            status, error_code = "not_found", "TARGET_NOT_FOUND"
        else:
            eligible[0]["selected"] = True
            warnings.extend(str(reason) for reason in eligible[0]["rejection_reasons"])
            if len(eligible) > 1:
                warnings.append("multiple_candidates_best_selected")
    else:
        if not accepted:
            status, error_code = "not_found", "TARGET_NOT_FOUND"
        elif len(accepted) > 1 and float(accepted[0]["confidence"]) - float(accepted[1]["confidence"]) < 0.15:
            status, error_code = "ambiguous", "AMBIGUOUS_TARGET"
        else:
            accepted[0]["selected"] = True
    return {
        "color": color,
        "selection_policy": selection_policy,
        "roi": [x1, y1, x2, y2],
        "status": status,
        "error_code": error_code,
        "warnings": warnings,
        "candidate_count": len(observations),
        "eligible_candidate_count": len(eligible) if selection_policy == "best_effort" else len(accepted),
        "ignored_tiny_fragment_count": len(observations) - len(eligible) if selection_policy == "best_effort" else 0,
        "fragment_area_floor_px": fragment_area_floor,
        "candidates": observations[:50],
        "selected": eligible[0] if selection_policy == "best_effort" and status == "success" else (accepted[0] if status == "success" else None),
    }


def build_detection_diagnostic(
    image_bgr: np.ndarray,
    *,
    profile: dict[str, Any],
    colors: tuple[str, ...],
    calibration_mode: bool = False,
    selection_policy: str = "strict",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw ROI and accepted/rejected boxes without changing recognition decisions."""

    image = np.asarray(image_bgr)
    config = _single_color_calibration_detector(image, profile) if calibration_mode else profile.get("detector")
    if not isinstance(config, dict):
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：颜色检测配置缺失。")
    reports = [analyze_color_candidates(image, color, config, selection_policy=selection_policy) for color in colors]
    marked = image.copy()
    # Tiny disconnected regions on a correctly detected block are surface/edge
    # fragments. Keep them in the report for analysis, but do not clutter the
    # diagnostic image with REJECT boxes and overlapping labels.
    diagnostic_reject_area_floor = float(config["min_area_px"]) * 0.05
    for report in reports:
        report_color = str(report["color"])
        x1, y1, x2, y2 = report["roi"]
        cv2.rectangle(marked, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2, cv2.LINE_AA)
        for candidate in report["candidates"]:
            if candidate.get("ignored_as_tiny_fragment"):
                continue
            if not candidate["selected"] and not candidate["accepted"] and float(candidate["area_px"]) < diagnostic_reject_area_floor:
                continue
            points = np.asarray(candidate["box"], dtype=np.int32).reshape((-1, 1, 2))
            if candidate["selected"]:
                box_color, state = (0, 220, 0), "RETURN" if selection_policy == "best_effort" else "OK"
            elif selection_policy == "best_effort":
                box_color, state = (0, 215, 255), "ALT"
            elif candidate["accepted"]:
                box_color, state = (0, 215, 255), "AMB"
            else:
                box_color, state = (0, 0, 255), "REJECT"
            cv2.polylines(marked, [points], True, box_color, 3 if candidate["selected"] else 2, cv2.LINE_AA)
            cx, cy = candidate["center"]
            label = f"{COLOR_LABELS[report_color]} {state} A={candidate['area_px']:.0f} C={candidate['confidence']:.2f}"
            cv2.putText(marked, label, (max(0, int(cx) - 60), max(18, int(cy) - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
    summary = {"success": all(report["status"] == "success" for report in reports), "colors": reports}
    return marked, summary


def _pixel_candidate(
    image: np.ndarray,
    color: str,
    config: dict[str, Any],
    *,
    selection_policy: str = "strict",
) -> tuple[float, tuple[float, float], float]:
    report = analyze_color_candidates(
        image, color, config, selection_policy=selection_policy,
    )
    if report["status"] == "not_found":
        raise WorkspaceRecognitionError("TARGET_NOT_FOUND", f"识别失败：未找到唯一有效的{color}色目标。")
    if report["status"] == "ambiguous":
        raise WorkspaceRecognitionError("AMBIGUOUS_TARGET", f"识别失败：{color}色存在多个相近候选目标。")
    selected = report["selected"]
    assert isinstance(selected, dict)
    return float(selected["confidence"]), (float(selected["center"][0]), float(selected["center"][1])), float(selected["angle_deg"])


def _normalize_square(angle_deg: float) -> float:
    return ((float(angle_deg) + 45.0) % 90.0) - 45.0


def estimate_six_color_area_range(image: np.ndarray, *, profile: dict[str, Any]) -> dict[str, Any]:
    """不依赖已有面积阈值，从同一真实帧估算六色方形目标的面积范围。"""

    detector = profile.get("detector")
    if not isinstance(detector, dict):
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：颜色检测配置缺失。")
    hsv_ranges = detector.get("hsv_ranges")
    try:
        confidence_min = float(detector.get("confidence_min", 0.6))
    except (TypeError, ValueError) as exc:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：置信度配置无效。") from exc
    if not isinstance(hsv_ranges, dict) or not 0 <= confidence_min <= 1:
        raise WorkspaceRecognitionError("DETECTOR_INVALID", "识别失败：HSV或置信度配置无效。")
    height, width = image.shape[:2]
    # The estimator deliberately ignores the saved detector area limits, but it
    # must not let a handful of square-shaped noise pixels outrank a real part.
    # Real calibration targets occupy a meaningful fraction of the full frame;
    # these broad bounds remain independent of the old configured thresholds.
    frame_area = float(width * height)
    coarse_min_area = max(9.0, frame_area * 0.001)
    coarse_max_area = frame_area * 0.10
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    areas: dict[str, float] = {}
    for color in ("红", "橙", "黄", "绿", "蓝", "紫"):
        ranges = hsv_ranges.get(color)
        if not isinstance(ranges, list) or not ranges:
            raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色HSV范围缺失。")
        mask = np.zeros((height, width), dtype=np.uint8)
        for band in ranges:
            try:
                lower = np.asarray(band["lower"], dtype=np.uint8)
                upper = np.asarray(band["upper"], dtype=np.uint8)
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色HSV范围无效。") from exc
            if lower.shape != (3,) or upper.shape != (3,):
                raise WorkspaceRecognitionError("DETECTOR_INVALID", f"识别失败：{color}色HSV范围必须是三通道。")
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < coarse_min_area or area > coarse_max_area:
                continue
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
            if min(rect_w, rect_h) < 3 or max(rect_w, rect_h) <= 0:
                continue
            aspect = min(rect_w, rect_h) / max(rect_w, rect_h)
            rectangularity = min(1.0, max(0.0, area / (rect_w * rect_h)))
            confidence = float(0.65 * aspect + 0.35 * rectangularity)
            if aspect >= 0.65 and confidence >= confidence_min:
                candidates.append((confidence, area))
        candidates.sort(reverse=True)
        if not candidates:
            raise WorkspaceRecognitionError("TARGET_NOT_FOUND", f"识别失败：自动面积估算未找到{color}色方形目标。")
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.15:
            raise WorkspaceRecognitionError("AMBIGUOUS_TARGET", f"识别失败：自动面积估算发现多个相近的{color}色目标。")
        areas[color] = candidates[0][1]
    measured_min, measured_max = min(areas.values()), max(areas.values())
    minimum = max(1.0, float(math.floor(measured_min * 0.5)))
    maximum = min(float(width * height), float(math.ceil(measured_max * 1.5)))
    if not minimum < maximum:
        raise WorkspaceRecognitionError("AREA_ESTIMATE_INVALID", "识别失败：自动面积范围无效。")
    return {"areas_px": areas, "min_area_px": minimum, "max_area_px": maximum}
