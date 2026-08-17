from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .paths import portable_project_path


SCENES = ("blocks", "trays")
SCENE_LABELS = {"blocks": "方块", "trays": "托盘"}
GRID_FACTORS = ((-1, -1), (0, -1), (1, -1), (1, 0), (0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1))
MAX_RESIDUAL_MM = 0.5
MAX_RMS_MM = 0.3
XY_VALIDATION_TARGET_MM = 10.0
XY_VALIDATION_TOLERANCE_MM = 0.5
ANGLE_ZERO_TOLERANCE_DEG = 2.0
ANGLE_TEN_MIN_DEG = 8.0
ANGLE_TEN_MAX_DEG = 12.0


class NinePointError(ValueError):
    pass


@dataclass(frozen=True)
class GridPoint:
    index: int
    camera_x_mm: float
    camera_y_mm: float

    @property
    def expected_tool_x_mm(self) -> float:
        return -self.camera_x_mm

    @property
    def expected_tool_y_mm(self) -> float:
        return -self.camera_y_mm


@dataclass(frozen=True)
class FitResult:
    homography: tuple[tuple[float, float, float], ...]
    residuals_mm: tuple[float, ...]
    rms_error_mm: float
    max_error_mm: float


def build_grid(step_x_mm: Any, step_y_mm: Any) -> tuple[GridPoint, ...]:
    sx, sy = _finite(step_x_mm, "X步长"), _finite(step_y_mm, "Y步长")
    if not (0.1 <= sx <= 100.0 and 0.1 <= sy <= 100.0):
        raise NinePointError("九点 X/Y步长必须在 0.1～100.0 mm。")
    return tuple(GridPoint(index, fx * sx, fy * sy) for index, (fx, fy) in enumerate(GRID_FACTORS, 1))


def fit_pixel_to_tool(samples: Sequence[Mapping[str, Any]]) -> FitResult:
    if len(samples) != 9:
        raise NinePointError("九点拟合必须恰好包含9个样本。")
    indexes = [sample.get("index") for sample in samples]
    if indexes != list(range(1, 10)):
        raise NinePointError("九点样本必须严格按1～9排列。")
    try:
        pixels = np.asarray([[float(sample["pixel_u"]), float(sample["pixel_v"])] for sample in samples], dtype=np.float64)
        tools = np.asarray([[float(sample["tool_x_mm"]), float(sample["tool_y_mm"])] for sample in samples], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise NinePointError("九点样本缺少有效像素或工具坐标。") from exc
    if not np.isfinite(pixels).all() or not np.isfinite(tools).all():
        raise NinePointError("九点样本包含非有限数值。")
    if np.ptp(pixels[:, 0]) < 5 or np.ptp(pixels[:, 1]) < 5 or np.ptp(tools[:, 0]) < 0.1 or np.ptp(tools[:, 1]) < 0.1:
        raise NinePointError("九点覆盖范围过小，无法可靠拟合。")
    matrix, _ = cv2.findHomography(pixels, tools, method=0)
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise NinePointError("九点单应矩阵拟合失败。")
    projected = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected - tools, axis=1)
    rms = float(math.sqrt(float(np.mean(np.square(residuals)))))
    maximum = float(np.max(residuals))
    if maximum > MAX_RESIDUAL_MM or rms > MAX_RMS_MM:
        worst = int(np.argmax(residuals)) + 1
        raise NinePointError(f"九点拟合误差超限：RMS={rms:.3f} mm，最大={maximum:.3f} mm（第{worst}点）。")
    return FitResult(
        tuple(tuple(float(value) for value in row) for row in matrix),
        tuple(float(value) for value in residuals), rms, maximum,
    )


def calibration_id(scene: str, camera_serial: str, samples: Sequence[Mapping[str, Any]]) -> str:
    if scene not in SCENES:
        raise NinePointError("标定场景必须是 blocks或trays。")
    material = json.dumps({"scene": scene, "camera_serial": camera_serial, "samples": list(samples)}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12].upper()
    return f"N9_{scene.upper()}_{suffix}"


def build_candidate(
    *, scene: str, camera_serial: str, robot_serial: str, active_tcp: str,
    photo_point: str, image_width: int, image_height: int,
    target_color: str, step_x_mm: float, step_y_mm: float,
    samples: Sequence[Mapping[str, Any]], fit: FitResult,
    reference_detections: Mapping[str, Mapping[str, Any]] | None = None,
    reference_image_path: str | None = None,
    reference_annotated_image_path: str | None = None,
) -> dict[str, Any]:
    if scene not in SCENES or photo_point != f"{scene}_photo":
        raise NinePointError("标定场景与拍照点不匹配。")
    for label, value in (("相机序列号", camera_serial), ("机器人序列号", robot_serial), ("活动TCP", active_tcp)):
        if not isinstance(value, str) or not value.strip() or value == "UNSET":
            raise NinePointError(f"{label}无效。")
    if not isinstance(image_width, int) or not isinstance(image_height, int) or min(image_width, image_height) <= 0:
        raise NinePointError("标定分辨率无效。")
    stored_samples = []
    for sample, residual in zip(samples, fit.residuals_mm):
        item = dict(sample)
        for field in ("image_path", "raw_image_path", "annotated_image_path"):
            if isinstance(item.get(field), str):
                item[field] = portable_project_path(item[field])
        item["residual_mm"] = residual
        stored_samples.append(item)
    cid = calibration_id(scene, camera_serial, stored_samples)
    references: dict[str, dict[str, float]] = {}
    if reference_detections is not None:
        if set(reference_detections) != {"红", "橙", "黄", "绿", "蓝", "紫"}:
            raise NinePointError("六色基准检测必须唯一覆盖红、橙、黄、绿、蓝、紫。")
        for color, item in reference_detections.items():
            try:
                values = {
                    "pixel_u": float(item["pixel_u"]),
                    "pixel_v": float(item["pixel_v"]),
                    "r_image_deg": float(item["r_image_deg"]),
                    "confidence": float(item["confidence"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise NinePointError(f"{color}色基准检测字段无效。") from exc
            if not all(math.isfinite(value) for value in values.values()) or not 0 <= values["confidence"] <= 1:
                raise NinePointError(f"{color}色基准检测数值无效。")
            references[color] = values
    result = {
        "schema_version": 1,
        "scene": scene,
        "data_origin": "camera_vision",
        "usable_for_real_robot": False,
        "approved": False,
        "calibration_id": cid,
        "camera_serial": camera_serial,
        "robot_serial": robot_serial,
        "active_tcp": active_tcp,
        "photo_point": photo_point,
        "image_width": image_width,
        "image_height": image_height,
        "target_color": target_color,
        "step_x_mm": float(step_x_mm),
        "step_y_mm": float(step_y_mm),
        "homography_pixel_to_tool_mm": [list(row) for row in fit.homography],
        "rms_error_mm": fit.rms_error_mm,
        "max_error_mm": fit.max_error_mm,
        "samples": stored_samples,
        "reference_detections": references,
        "reference_image_path": portable_project_path(reference_image_path) if reference_image_path else None,
        "reference_annotated_image_path": portable_project_path(reference_annotated_image_path) if reference_annotated_image_path else None,
        "direction_validation": {
            "x_positive": None, "y_positive": None,
            "angle_zero": None, "angle_positive_10deg": None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approved_at": None,
    }
    return result


def validate_direction_results(validation: Mapping[str, Any]) -> None:
    if validation.get("skipped") is True and validation.get("reason") == "operator_requested_direct_activation_after_nine_point":
        return
    try:
        x = validation["x_positive"]
        y = validation["y_positive"]
        angle_zero = float(validation["angle_zero"])
        angle_ten = float(validation["angle_positive_10deg"])
        x_dx, x_dy = float(x["dx_mm"]), float(x["dy_mm"])
        y_dx, y_dy = float(y["dx_mm"]), float(y["dy_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NinePointError("方向验证数据不完整。") from exc
    if abs(x_dx - XY_VALIDATION_TARGET_MM) > XY_VALIDATION_TOLERANCE_MM or abs(x_dy) > XY_VALIDATION_TOLERANCE_MM:
        raise NinePointError("工具 +X 10 mm方向验证未通过。")
    if abs(y_dy - XY_VALIDATION_TARGET_MM) > XY_VALIDATION_TOLERANCE_MM or abs(y_dx) > XY_VALIDATION_TOLERANCE_MM:
        raise NinePointError("工具 +Y 10 mm方向验证未通过。")
    if abs(angle_zero) > ANGLE_ZERO_TOLERANCE_DEG or not ANGLE_TEN_MIN_DEG <= angle_ten <= ANGLE_TEN_MAX_DEG:
        raise NinePointError("0°或+10°角度验证未通过。")


def approve_candidate(candidate_path: Path, active_path: Path, archive_root: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _read_json(candidate_path)
    validate_direction_results(validation)
    return _activate_candidate(candidate, active_path, archive_root, dict(validation), "direction_validated")


def approve_candidate_without_direction_validation(candidate_path: Path, active_path: Path, archive_root: Path) -> dict[str, Any]:
    candidate = _read_json(candidate_path)
    skipped = {
        "skipped": True,
        "reason": "operator_requested_direct_activation_after_nine_point",
    }
    return _activate_candidate(candidate, active_path, archive_root, skipped, "operator_skipped_direction_validation")


def _activate_candidate(candidate: Mapping[str, Any], active_path: Path, archive_root: Path, validation: Mapping[str, Any], approval_source: str) -> dict[str, Any]:
    if candidate.get("approved") is True or candidate.get("usable_for_real_robot") is True:
        raise NinePointError("候选文件状态无效。")
    if float(candidate.get("max_error_mm", math.inf)) > MAX_RESIDUAL_MM or float(candidate.get("rms_error_mm", math.inf)) > MAX_RMS_MM:
        raise NinePointError("候选九点误差超限。")
    references = candidate.get("reference_detections")
    if not isinstance(references, dict) or set(references) != {"红", "橙", "黄", "绿", "蓝", "紫"}:
        raise NinePointError("候选九点缺少中心拍照位的六色基准检测，禁止启用。")
    approved = dict(candidate)
    approved["direction_validation"] = dict(validation)
    approved["approval_source"] = approval_source
    approved["approved"] = True
    approved["usable_for_real_robot"] = True
    approved["approved_at"] = datetime.now(timezone.utc).isoformat()
    active_path.parent.mkdir(parents=True, exist_ok=True)
    if active_path.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(active_path, archive_root / f"{active_path.stem}-{timestamp}.json")
    _write_json_atomic(active_path, approved)
    return approved


def write_candidate(path: Path, value: Mapping[str, Any]) -> None:
    _write_json_atomic(path, dict(value))


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NinePointError(f"{label}不是数值。") from exc
    if not math.isfinite(number):
        raise NinePointError(f"{label}不是有限数值。")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NinePointError(f"标定文件无法读取：{exc}") from exc
    if not isinstance(value, dict):
        raise NinePointError("标定文件顶层不是对象。")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
