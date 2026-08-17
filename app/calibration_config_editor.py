from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .nine_point import SCENES, build_grid


COLORS = ("红", "橙", "黄", "绿", "蓝", "紫")


class CalibrationConfigError(ValueError):
    pass


def load_calibration_settings(path: Path) -> dict[str, Any]:
    motion = _read(path)
    raw = motion.get("nine_point")
    if not isinstance(raw, dict):
        raise CalibrationConfigError("motion.json缺少 nine_point配置。")
    return json.loads(json.dumps(raw, ensure_ascii=False, allow_nan=False))


def save_calibration_settings(
    path: Path, *, linear_acceleration_m_s2: Any, linear_velocity_m_s: Any,
    settle_s: Any, scenes: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    if set(scenes) != set(SCENES):
        raise CalibrationConfigError("blocks和trays参数必须全部提供。")
    acceleration = _positive(linear_acceleration_m_s2, "九点直线加速度")
    velocity = _positive(linear_velocity_m_s, "九点直线速度")
    settle = _positive(settle_s, "稳定等待")
    if acceleration > 10 or velocity > 2 or not 0.2 <= settle <= 3.0:
        raise CalibrationConfigError("九点速度、加速度或稳定时间超出保护范围。")
    normalized: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        value = scenes[scene]
        color = str(value.get("target_color", "")).strip()
        if color not in COLORS:
            raise CalibrationConfigError(f"{scene}目标颜色无效。")
        grid = build_grid(value.get("step_x_mm"), value.get("step_y_mm"))
        normalized[scene] = {
            "step_x_mm": abs(grid[0].camera_x_mm),
            "step_y_mm": abs(grid[0].camera_y_mm),
            "target_color": color,
        }
    motion = _read(path)
    old = motion.get("nine_point", {})
    changed_scenes = {
        scene for scene in SCENES
        if not isinstance(old.get(scene), dict)
        or any(old[scene].get(key) != normalized[scene][key] for key in ("step_x_mm", "step_y_mm", "target_color"))
    }
    nine = motion.setdefault("nine_point", {})
    nine.update({
        "speed_fraction": 0.5,
        "linear_acceleration_m_s2": acceleration,
        "linear_velocity_m_s": velocity,
        "settle_s": settle,
    })
    for scene in SCENES:
        verified = bool(old.get(scene, {}).get("automatic_verified")) if scene not in changed_scenes else False
        nine[scene] = {**normalized[scene], "automatic_verified": verified}
    _write(path, motion)
    return changed_scenes


def mark_automatic_verified(path: Path, scene: str, verified: bool) -> None:
    if scene not in SCENES:
        raise CalibrationConfigError("场景无效。")
    motion = _read(path)
    try:
        motion["nine_point"][scene]["automatic_verified"] = bool(verified)
    except (KeyError, TypeError) as exc:
        raise CalibrationConfigError("九点配置结构无效。") from exc
    _write(path, motion)


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationConfigError(f"{label}不是数值。") from exc
    if not math.isfinite(number) or number <= 0:
        raise CalibrationConfigError(f"{label}必须是正有限数。")
    return number


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationConfigError(f"motion.json无法读取：{exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CalibrationConfigError("motion.json版本无效。")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
