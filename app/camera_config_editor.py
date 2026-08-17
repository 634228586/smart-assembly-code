from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROFILE_KEYS = ("task_card", "blocks", "trays")
PROFILE_LABELS = {
    "task_card": "任务卡",
    "blocks": "方块",
    "trays": "托盘",
}
DETECTOR_COLORS = ("红", "橙", "黄", "绿", "蓝", "紫")
SERIAL_NUMBER = re.compile(r"[A-Za-z0-9_.\-]{1,128}\Z")


class CameraConfigInputError(ValueError):
    pass


def load_camera_editor_values(path: Path) -> tuple[str, dict[str, dict[str, float | int | None]]]:
    camera = _read_json(path)
    serial = "" if camera.get("serial_number") == "UNSET" else str(camera.get("serial_number", ""))
    profiles: dict[str, dict[str, float | int | None]] = {}
    raw_profiles = camera.get("profiles", {})
    for key in PROFILE_KEYS:
        raw = raw_profiles.get(key, {}) if isinstance(raw_profiles, dict) else {}
        white = raw.get("white_balance", {}) if isinstance(raw, dict) else {}
        roi = raw.get("roi", {}) if isinstance(raw, dict) else {}
        profiles[key] = {
            "exposure_us": _optional_float(raw.get("exposure_us")),
            "gain": _optional_float(raw.get("gain")),
            "white_red": _optional_float(white.get("red") if isinstance(white, dict) else None),
            "white_green": _optional_float(white.get("green") if isinstance(white, dict) else None),
            "white_blue": _optional_float(white.get("blue") if isinstance(white, dict) else None),
            "width": _optional_int(roi.get("width") if isinstance(roi, dict) else None),
            "height": _optional_int(roi.get("height") if isinstance(roi, dict) else None),
            "offset_x": _optional_int(roi.get("offset_x") if isinstance(roi, dict) else None),
            "offset_y": _optional_int(roi.get("offset_y") if isinstance(roi, dict) else None),
        }
    return serial, profiles


def save_camera_editor_values(path: Path, *, serial_number: str, profile_values: Mapping[str, Mapping[str, Any]]) -> None:
    serial = str(serial_number).strip()
    if SERIAL_NUMBER.fullmatch(serial) is None:
        raise CameraConfigInputError("相机序列号必须是 1 至 128 个字母、数字、点、下划线或连字符；请从当前真实 MVS设备信息抄写。")
    if set(profile_values) != set(PROFILE_KEYS):
        raise CameraConfigInputError("task_card、blocks、trays 三套采集参数必须全部填写。")

    canonical: dict[str, dict[str, Any]] = {}
    required = {"exposure_us", "gain", "white_red", "white_green", "white_blue", "width", "height", "offset_x", "offset_y"}
    for key in PROFILE_KEYS:
        values = profile_values[key]
        if set(values) != required:
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}采集参数字段不完整。")
        exposure = _finite(values["exposure_us"], f"{PROFILE_LABELS[key]}曝光")
        gain = _finite(values["gain"], f"{PROFILE_LABELS[key]}增益")
        white = [_finite(values[name], f"{PROFILE_LABELS[key]}白平衡") for name in ("white_red", "white_green", "white_blue")]
        width, height = (_integer(values[name], f"{PROFILE_LABELS[key]}分辨率") for name in ("width", "height"))
        offset_x, offset_y = (_integer(values[name], f"{PROFILE_LABELS[key]}偏移") for name in ("offset_x", "offset_y"))
        if not (0 < exposure <= 10_000_000):
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}曝光必须大于 0 且不超过 10000000 us。")
        if not (0 <= gain <= 1000):
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}增益必须在 0 至 1000 之间。")
        if any(not 0 < value <= 65535 for value in white):
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}白平衡 R/G/B 必须大于 0 且不超过 65535。")
        if not (0 < width <= 100_000 and 0 < height <= 100_000 and 0 <= offset_x <= 100_000 and 0 <= offset_y <= 100_000):
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}宽高或 Offset 数值无效。")
        canonical[key] = {
            "exposure_us": exposure,
            "gain": gain,
            "white_balance": {"red": white[0], "green": white[1], "blue": white[2]},
            "roi": {"width": width, "height": height, "offset_x": offset_x, "offset_y": offset_y},
        }

    camera = _read_json(path)
    if camera.get("schema_version") != 1 or camera.get("sdk_family") != "hikrobot_mvs":
        raise CameraConfigInputError("camera.json版本或 SDK类型无效。")
    profiles = camera.get("profiles")
    if not isinstance(profiles, dict) or set(PROFILE_KEYS) - set(profiles):
        raise CameraConfigInputError("camera.json缺少三套正式 profile。")
    camera["serial_number"] = serial
    for key in PROFILE_KEYS:
        profile = profiles[key]
        if not isinstance(profile, dict):
            raise CameraConfigInputError(f"{key} profile不是对象。")
        profile.update(canonical[key])
        profile["trigger_mode"] = "software"
        # 修改参数会撤销旧批准；界面在操作者明确确认后立即重新批准。
        profile["approved"] = False
        detector = profile.get("detector")
        if isinstance(detector, dict):
            detector["roi"] = [0, 0, canonical[key]["roi"]["width"], canonical[key]["roi"]["height"]]
            detector["approved"] = False
    _write_json_atomic(path, camera)


def load_detector_editor_values(path: Path) -> dict[str, dict[str, Any]]:
    camera = _read_json(path)
    result: dict[str, dict[str, Any]] = {}
    for scene in ("blocks", "trays"):
        detector = camera.get("profiles", {}).get(scene, {}).get("detector")
        if not isinstance(detector, dict):
            raise CameraConfigInputError(f"{scene} detector配置缺失。")
        result[scene] = json.loads(json.dumps(detector, ensure_ascii=False, allow_nan=False))
    return result


def save_detector_editor_values(path: Path, *, scene: str, roi_values: Sequence[Any], confidence_min: Any, min_area_px: Any, max_area_px: Any, hsv_json: str) -> bool:
    if scene not in {"blocks", "trays"}:
        raise CameraConfigInputError("颜色参数场景无效。")
    confidence = _finite(confidence_min, "置信度")
    min_area, max_area = _finite(min_area_px, "最小面积"), _finite(max_area_px, "最大面积")
    if not (0 <= confidence <= 1 and 0 < min_area < max_area):
        raise CameraConfigInputError("置信度或面积范围无效。")
    try:
        hsv = json.loads(hsv_json)
    except json.JSONDecodeError as exc:
        raise CameraConfigInputError(f"HSV JSON无效：{exc}") from exc
    _validate_hsv(hsv)
    camera = _read_json(path)
    profile = camera.get("profiles", {}).get(scene)
    if not isinstance(profile, dict) or not isinstance(profile.get("detector"), dict):
        raise CameraConfigInputError(f"{scene} detector配置缺失。")
    image_roi = profile.get("roi", {})
    try:
        width, height = int(image_roi["width"]), int(image_roi["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CameraConfigInputError("必须先填写本场景采集分辨率。") from exc
    if width <= 0 or height <= 0:
        raise CameraConfigInputError("本场景采集分辨率无效。")
    # 正式比赛固定使用完整采集画面，避免现场误填检测裁剪范围。
    roi = [0, 0, width, height]
    detector = profile["detector"]
    saved_values = {
        "roi": roi, "confidence_min": confidence,
        "min_area_px": min_area, "max_area_px": max_area, "hsv_ranges": hsv,
    }
    changed = any(detector.get(key) != value for key, value in saved_values.items())
    detector.update(saved_values)
    # Re-saving byte-for-byte equivalent detector values must not destroy a
    # fresh real-frame approval. Any actual parameter change still fails safe.
    if changed:
        detector["approved"] = False
    _write_json_atomic(path, camera)
    return changed


def approve_detector_values(path: Path, *, scene: str, expected_sha256: str) -> None:
    camera = _read_json(path)
    try:
        detector = camera["profiles"][scene]["detector"]
    except (KeyError, TypeError) as exc:
        raise CameraConfigInputError("颜色参数配置缺失。") from exc
    actual = hashlib.sha256(json.dumps(detector, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if actual != expected_sha256:
        raise CameraConfigInputError("颜色参数在真实六色验证后发生变化，拒绝批准。")
    detector["approved"] = True
    _write_json_atomic(path, camera)


def approve_profile_values(path: Path, *, profile_name: str, expected_sha256: str) -> None:
    approve_profile_batch(path, expected_sha256_by_profile={profile_name: expected_sha256})


def approve_profile_batch_by_operator(path: Path) -> None:
    """Approve all saved profiles from an explicit on-site operator confirmation."""

    camera = _read_json(path)
    profiles = camera.get("profiles")
    if not isinstance(profiles, dict) or set(PROFILE_KEYS) - set(profiles):
        raise CameraConfigInputError("camera.json缺少三套正式 profile。")
    for key in PROFILE_KEYS:
        profile = profiles[key]
        if not isinstance(profile, dict) or profile.get("trigger_mode") != "software":
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]} profile无效或不是软件触发。")
        required = {
            "exposure_us": profile.get("exposure_us"), "gain": profile.get("gain"),
            "white_balance": profile.get("white_balance"), "roi": profile.get("roi"),
        }
        if _contains_unset(required):
            raise CameraConfigInputError(f"{PROFILE_LABELS[key]}采集参数尚未填写完整。")
        profile["approved"] = True
        profile["approval_source"] = "operator_confirmed_without_preflight_readback"
    _write_json_atomic(path, camera)


def approve_profile_batch(path: Path, *, expected_sha256_by_profile: Mapping[str, str]) -> None:
    if not expected_sha256_by_profile or set(expected_sha256_by_profile) - set(PROFILE_KEYS):
        raise CameraConfigInputError("采集参数 profile名称无效。")
    camera = _read_json(path)
    validated: list[dict[str, Any]] = []
    for profile_name, expected_sha256 in expected_sha256_by_profile.items():
        try:
            profile = camera["profiles"][profile_name]
        except (KeyError, TypeError) as exc:
            raise CameraConfigInputError("采集参数 profile缺失。") from exc
        actual = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if actual != expected_sha256:
            raise CameraConfigInputError(f"{profile_name}采集参数在写入测试后发生变化，拒绝批准。")
        validated.append(profile)
    for profile in validated:
        profile["approved"] = True
    _write_json_atomic(path, camera)


def _contains_unset(value: Any) -> bool:
    if value in (None, "UNSET"):
        return True
    if isinstance(value, dict):
        return any(_contains_unset(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unset(item) for item in value)
    return False


def _validate_hsv(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(DETECTOR_COLORS):
        raise CameraConfigInputError("HSV必须完整且仅包含红、橙、黄、绿、蓝、紫六色。")
    for color in DETECTOR_COLORS:
        bands = value[color]
        if not isinstance(bands, list) or not bands:
            raise CameraConfigInputError(f"{color}色 HSV范围缺失。")
        for band in bands:
            if not isinstance(band, dict) or set(band) != {"lower", "upper"}:
                raise CameraConfigInputError(f"{color}色 HSV区间字段无效。")
            lower, upper = band["lower"], band["upper"]
            if not isinstance(lower, list) or not isinstance(upper, list) or len(lower) != 3 or len(upper) != 3:
                raise CameraConfigInputError(f"{color}色 HSV上下限必须各有三项。")
            lo = [_integer(item, f"{color}色HSV") for item in lower]
            hi = [_integer(item, f"{color}色HSV") for item in upper]
            limits = (179, 255, 255)
            if any(not 0 <= lo[i] <= hi[i] <= limits[i] for i in range(3)):
                raise CameraConfigInputError(f"{color}色 HSV范围越界或上下限颠倒。")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraConfigInputError(f"配置文件无法读取：{path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise CameraConfigInputError(f"配置顶层不是对象：{path.name}")
    return value


def _optional_float(value: Any) -> float | None:
    if value in (None, "UNSET"):
        return None
    try:
        return _finite(value, "配置值")
    except CameraConfigInputError:
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, "UNSET"):
        return None
    try:
        return _integer(value, "配置值")
    except CameraConfigInputError:
        return None


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraConfigInputError(f"{label}不是数值。") from exc
    if not math.isfinite(number):
        raise CameraConfigInputError(f"{label}不是有限数值。")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CameraConfigInputError(f"{label}不是整数。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraConfigInputError(f"{label}不是整数。") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise CameraConfigInputError(f"{label}不是整数。")
    return int(number)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
