from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.paths import PACKAGE_ROOT, portable_project_path
from .workspace_localizer import COLORS, WorkspaceRecognitionError, build_detection_diagnostic


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_COLORS_BGR = {
    "红": (0, 0, 255),
    "橙": (0, 140, 255),
    "黄": (0, 255, 255),
    "绿": (0, 180, 0),
    "蓝": (255, 80, 0),
    "紫": (180, 0, 180),
}


class OfflineAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfflineImageResult:
    path: Path
    original_bgr: np.ndarray
    mask_bgr: np.ndarray
    annotated_bgr: np.ndarray
    returned_bgr: np.ndarray
    summary: dict[str, Any]


def discover_images(paths: Iterable[str | Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.casefold() in IMAGE_EXTENSIONS:
                if candidate.stem.casefold().endswith("-annotated"):
                    raw_name = candidate.stem[:-len("-annotated")] + candidate.suffix
                    raw_candidate = candidate.with_name(raw_name)
                    if raw_candidate.is_file():
                        candidate = raw_candidate
                resolved = candidate.resolve()
                found[str(resolved).casefold()] = resolved
    return sorted(found.values(), key=lambda value: str(value).casefold())


def read_image(path: str | Path) -> np.ndarray:
    value = Path(path)
    try:
        payload = np.fromfile(value, dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise OfflineAnalysisError(f"图片读取失败：{value}：{exc}") from exc
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise OfflineAnalysisError(f"图片不是有效的BGR彩色图像：{value}")
    return np.ascontiguousarray(image)


def normalize_detector_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise OfflineAnalysisError("检测参数必须是JSON对象。")
    try:
        roi = [int(value) for value in raw["roi"]]
        confidence = float(raw["confidence_min"])
        minimum = float(raw["min_area_px"])
        maximum = float(raw["max_area_px"])
        hsv_ranges = raw["hsv_ranges"]
    except (KeyError, TypeError, ValueError) as exc:
        raise OfflineAnalysisError("检测参数缺少ROI、面积、置信度或HSV字段。") from exc
    if len(roi) != 4 or not 0 <= confidence <= 1 or not 0 < minimum < maximum:
        raise OfflineAnalysisError("ROI、面积范围或置信度无效。")
    if not isinstance(hsv_ranges, dict) or set(hsv_ranges) != set(COLORS):
        raise OfflineAnalysisError("HSV必须完整包含红、橙、黄、绿、蓝、紫六种颜色。")
    normalized_hsv: dict[str, list[dict[str, list[int]]]] = {}
    for color in COLORS:
        bands = hsv_ranges[color]
        if not isinstance(bands, list) or not bands:
            raise OfflineAnalysisError(f"{color}色至少需要一个HSV区间。")
        normalized_bands = []
        for band in bands:
            try:
                lower = [int(value) for value in band["lower"]]
                upper = [int(value) for value in band["upper"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise OfflineAnalysisError(f"{color}色HSV区间格式无效。") from exc
            if len(lower) != 3 or len(upper) != 3:
                raise OfflineAnalysisError(f"{color}色HSV上下限必须各有三个整数。")
            channel_max = (179, 255, 255)
            if any(not 0 <= lower[index] <= upper[index] <= channel_max[index] for index in range(3)):
                raise OfflineAnalysisError(f"{color}色HSV上下限越界或顺序错误。")
            normalized_bands.append({"lower": lower, "upper": upper})
        normalized_hsv[color] = normalized_bands
    return {
        "roi": roi,
        "confidence_min": confidence,
        "min_area_px": minimum,
        "max_area_px": maximum,
        "hsv_ranges": normalized_hsv,
    }


def load_scene_detector(camera_path: str | Path, scene: str) -> dict[str, Any]:
    if scene not in {"blocks", "trays"}:
        raise OfflineAnalysisError("场景必须是blocks或trays。")
    try:
        camera = json.loads(Path(camera_path).read_text(encoding="utf-8"))
        detector = camera["profiles"][scene]["detector"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OfflineAnalysisError(f"无法从camera.json读取{scene}检测参数。") from exc
    return normalize_detector_config(detector)


def _color_mask(image_bgr: np.ndarray, detector: dict[str, Any], color: str) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = detector["roi"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise OfflineAnalysisError(f"检测ROI {detector['roi']} 超出图片尺寸 {width}x{height}。")
    crop = cv2.GaussianBlur(image_bgr[y1:y2, x1:x2], (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    local_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for band in detector["hsv_ranges"][color]:
        local_mask = cv2.bitwise_or(
            local_mask,
            cv2.inRange(hsv, np.asarray(band["lower"], dtype=np.uint8), np.asarray(band["upper"], dtype=np.uint8)),
        )
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    result = np.zeros((height, width), dtype=np.uint8)
    result[y1:y2, x1:x2] = local_mask
    return result


def build_combined_mask(image_bgr: np.ndarray, detector: dict[str, Any]) -> np.ndarray:
    marked = np.zeros_like(image_bgr)
    for color in COLORS:
        mask = _color_mask(image_bgr, detector, color)
        marked[mask > 0] = MASK_COLORS_BGR[color]
    return marked


RETURN_LABELS = {"红": "RED", "橙": "ORANGE", "黄": "YELLOW", "绿": "GREEN", "蓝": "BLUE", "紫": "PURPLE"}


def build_returned_points_image(image_bgr: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    """Draw only the points actually returned by best-effort recognition."""

    marked = image_bgr.copy()
    for report in summary.get("colors", []):
        selected = report.get("selected") if isinstance(report, dict) else None
        if not isinstance(selected, dict):
            continue
        center = selected.get("center")
        if not isinstance(center, list) or len(center) != 2:
            continue
        color = str(report.get("color", ""))
        cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
        marker = MASK_COLORS_BGR.get(color, (255, 255, 255))
        # A black outline keeps the returned point visible on both bright paper
        # and a target whose fill color matches the marker.
        cv2.drawMarker(marked, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 54, 7, cv2.LINE_AA)
        cv2.drawMarker(marked, (cx, cy), marker, cv2.MARKER_CROSS, 48, 3, cv2.LINE_AA)
        cv2.circle(marked, (cx, cy), 14, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.circle(marked, (cx, cy), 14, marker, 3, cv2.LINE_AA)
        label = f"{RETURN_LABELS.get(color, color)} RETURN U={center[0]:.1f} V={center[1]:.1f}"
        origin = (min(max(5, cx + 22), max(5, marked.shape[1] - 330)), max(24, cy - 18))
        cv2.putText(marked, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(marked, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, marker, 2, cv2.LINE_AA)
    return marked


def analyze_image(path: str | Path, detector: dict[str, Any]) -> OfflineImageResult:
    normalized = normalize_detector_config(detector)
    image = read_image(path)
    profile = {"detector": normalized}
    try:
        annotated, summary = build_detection_diagnostic(
            image,
            profile=profile,
            colors=COLORS,
            selection_policy="best_effort",
        )
    except WorkspaceRecognitionError as exc:
        raise OfflineAnalysisError(f"检测失败[{exc.code}]：{exc}") from exc
    mask = build_combined_mask(image, normalized)
    returned = build_returned_points_image(image, summary)
    return OfflineImageResult(Path(path).resolve(), image, mask, annotated, returned, summary)


def write_candidate(
    output_path: str | Path,
    *,
    scene: str,
    detector: dict[str, Any],
    source_images: Iterable[str | Path],
) -> Path:
    if scene not in {"blocks", "trays"}:
        raise OfflineAnalysisError("候选参数场景必须是blocks或trays。")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "offline_detector_candidate",
        "scene": scene,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formal_camera_json_modified": False,
        "source_images": [portable_project_path(value) for value in source_images],
        "detector": normalize_detector_config(detector),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path.resolve()


def default_candidate_directory() -> Path:
    return PACKAGE_ROOT / "data" / "offline_vision_candidates"
