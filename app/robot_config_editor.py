from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


POINT_KEYS = ("competition_standby", "task_card_photo", "blocks_photo", "trays_photo")
COLORS = ("红", "橙", "黄", "绿", "蓝", "紫")
POINT_LABELS = {
    "competition_standby": "比赛待机点",
    "task_card_photo": "任务卡拍照点",
    "blocks_photo": "方块拍照点",
    "trays_photo": "托盘拍照点",
}
REFERENCE_LABELS = {"blocks": "Block基准抓取点", "trays": "Tray基准放置点"}
TCP_NAME = re.compile(r"[A-Za-z0-9_.\-\u4e00-\u9fff]{1,64}\Z")


class RobotConfigInputError(ValueError):
    pass


def load_robot_editor_values(robot_path: Path, motion_path: Path) -> tuple[str, list[float] | None, dict[str, list[float] | None]]:
    robot = _read_json(robot_path)
    motion = _read_json(motion_path)
    active_tcp = robot.get("active_tcp", {})
    name = "" if active_tcp.get("name") == "UNSET" else str(active_tcp.get("name", ""))
    offset = _optional_six(active_tcp.get("offset"))
    points = {key: _optional_six(motion.get("points", {}).get(key)) for key in POINT_KEYS}
    return name, offset, points


def save_robot_editor_values(
    robot_path: Path,
    motion_path: Path,
    *,
    tcp_name: str,
    tcp_values: Sequence[Any],
    tcp_units: str,
    point_values: Mapping[str, Sequence[Any]],
    joint_units: str,
) -> None:
    name = str(tcp_name).strip()
    if TCP_NAME.fullmatch(name) is None:
        raise RobotConfigInputError("活动 TCP名称必须为1至64个中文、字母、数字、点、下划线或连字符。")
    tcp = _six_finite(tcp_values, "活动 TCP offset")
    if tcp_units == "mm_deg":
        tcp = [value / 1000.0 for value in tcp[:3]] + [math.radians(value) for value in tcp[3:]]
    elif tcp_units != "m_rad":
        raise RobotConfigInputError("未知 TCP输入单位。")
    if max(abs(value) for value in tcp[:3]) > 2.0 or max(abs(value) for value in tcp[3:]) > 2 * math.pi:
        raise RobotConfigInputError("活动 TCP offset超出保护范围；请检查毫米/米或度/弧度是否选错。")

    canonical_points: dict[str, list[float]] = {}
    if set(point_values) != set(POINT_KEYS):
        raise RobotConfigInputError("四个关节点必须全部填写。")
    for key in POINT_KEYS:
        values = _six_finite(point_values[key], POINT_LABELS[key])
        if joint_units == "deg":
            values = [math.radians(value) for value in values]
        elif joint_units != "rad":
            raise RobotConfigInputError("未知关节输入单位。")
        if max(abs(value) for value in values) > 4 * math.pi:
            raise RobotConfigInputError(f"{POINT_LABELS[key]}超出保护范围；请检查度/弧度是否选错。")
        canonical_points[key] = values
    robot = _read_json(robot_path)
    motion = _read_json(motion_path)
    if robot.get("schema_version") != 1 or motion.get("schema_version") != 1:
        raise RobotConfigInputError("robot.json或motion.json版本无效。")
    previous_tcp = (robot.get("active_tcp", {}).get("name"), robot.get("active_tcp", {}).get("offset"))
    previous_points = dict(motion.get("points", {}))
    robot.setdefault("active_tcp", {})["name"] = name
    robot["active_tcp"]["offset"] = tcp
    motion.setdefault("points", {}).update(canonical_points)
    motion.pop("contact_z", None)
    if previous_tcp != (name, tcp):
        _invalidate_scene_geometry(motion, "blocks")
        _invalidate_scene_geometry(motion, "trays")
    else:
        if previous_points.get("blocks_photo") != canonical_points["blocks_photo"]:
            _invalidate_scene_geometry(motion, "blocks")
        if previous_points.get("trays_photo") != canonical_points["trays_photo"]:
            _invalidate_scene_geometry(motion, "trays")
    motion["real_robot_verified"] = False
    _replace_two_json(robot_path, robot, motion_path, motion)


def save_single_joint_point(motion_path: Path, *, point_key: str, joint_positions: Sequence[Any]) -> list[float]:
    """原子保存一个真实只读采集关节点，不要求其他点位或TCP已配置。"""

    if point_key not in POINT_KEYS:
        raise RobotConfigInputError("未知固定关节点。")
    canonical = _six_finite(joint_positions, POINT_LABELS[point_key])
    if max(abs(value) for value in canonical) > 4 * math.pi:
        raise RobotConfigInputError(f"{POINT_LABELS[point_key]}超出保护范围。")
    motion = _read_json(motion_path)
    if motion.get("schema_version") != 1:
        raise RobotConfigInputError("motion.json版本无效。")
    motion.setdefault("points", {})[point_key] = canonical
    motion["real_robot_verified"] = False
    if point_key == "blocks_photo":
        _invalidate_scene_geometry(motion, "blocks")
    if point_key == "trays_photo":
        _invalidate_scene_geometry(motion, "trays")
    temp = _write_temp(motion_path, motion)
    try:
        os.replace(temp, motion_path)
    finally:
        temp.unlink(missing_ok=True)
    return canonical


def load_reference_anchors(motion_path: Path) -> dict[str, dict[str, list[float] | None]]:
    motion = _read_json(motion_path)
    return normalize_reference_anchors(motion.get("reference_anchors", {}), strict=False)


def normalize_reference_anchors(value: Any, *, strict: bool = True) -> dict[str, dict[str, list[float] | None]]:
    """Return the six-colour shape, migrating a legacy scene pose to red only."""

    if not isinstance(value, dict):
        if strict:
            raise RobotConfigInputError("reference_anchors必须是对象。")
        value = {}
    result: dict[str, dict[str, list[float] | None]] = {}
    for scene in REFERENCE_LABELS:
        raw_scene = value.get(scene)
        if isinstance(raw_scene, (list, tuple)):
            # Legacy format: the sole physical pose was the red anchor.
            raw_scene = {"红": raw_scene}
        elif raw_scene is None or raw_scene == "UNSET":
            raw_scene = {}
        elif not isinstance(raw_scene, dict):
            if strict:
                raise RobotConfigInputError(f"{REFERENCE_LABELS[scene]}必须按颜色保存。")
            raw_scene = {}
        scene_result: dict[str, list[float] | None] = {}
        for color in COLORS:
            raw_pose = raw_scene.get(color, "UNSET")
            if raw_pose is None or raw_pose == "UNSET":
                scene_result[color] = None
            elif strict:
                scene_result[color] = _validated_reference_pose(raw_pose, scene, color)
            else:
                scene_result[color] = _optional_six(raw_pose)
        result[scene] = scene_result
    return result


def save_reference_anchor(motion_path: Path, *, scene: str, color: str = "红", tcp_pose: Sequence[Any]) -> list[float]:
    """Save one operator-taught colour-specific physical anchor; never moves the robot."""

    if scene not in REFERENCE_LABELS:
        raise RobotConfigInputError("未知基准抓取/放置场景。")
    if color not in COLORS:
        raise RobotConfigInputError("未知基准点颜色。")
    pose = _validated_reference_pose(tcp_pose, scene, color)
    motion = _read_json(motion_path)
    if motion.get("schema_version") != 1:
        raise RobotConfigInputError("motion.json版本无效。")
    anchors = normalize_reference_anchors(motion.get("reference_anchors", {}), strict=True)
    anchors[scene][color] = pose
    motion["reference_anchors"] = _serialize_reference_anchors(anchors)
    motion["real_robot_verified"] = False
    temp = _write_temp(motion_path, motion)
    try:
        os.replace(temp, motion_path)
    finally:
        temp.unlink(missing_ok=True)
    return pose


def clear_reference_anchor(motion_path: Path, *, scene: str) -> None:
    """Invalidate all six physical anchors after scene geometry/calibration changes."""

    if scene not in REFERENCE_LABELS:
        raise RobotConfigInputError("未知基准抓取/放置场景。")
    motion = _read_json(motion_path)
    if motion.get("schema_version") != 1:
        raise RobotConfigInputError("motion.json版本无效。")
    anchors = normalize_reference_anchors(motion.get("reference_anchors", {}), strict=False)
    anchors[scene] = {color: None for color in COLORS}
    motion["reference_anchors"] = _serialize_reference_anchors(anchors)
    motion["real_robot_verified"] = False
    temp = _write_temp(motion_path, motion)
    try:
        os.replace(temp, motion_path)
    finally:
        temp.unlink(missing_ok=True)


def _invalidate_scene_geometry(motion: dict[str, Any], scene: str) -> None:
    nine_point = motion.get("nine_point")
    if isinstance(nine_point, dict) and isinstance(nine_point.get(scene), dict):
        nine_point[scene]["automatic_verified"] = False
    anchors = normalize_reference_anchors(motion.get("reference_anchors", {}), strict=False)
    anchors[scene] = {color: None for color in COLORS}
    motion["reference_anchors"] = _serialize_reference_anchors(anchors)


def _validated_reference_pose(values: Sequence[Any], scene: str, color: str) -> list[float]:
    label = f"{REFERENCE_LABELS[scene]}({color})"
    pose = _six_finite(values, label)
    if max(abs(value) for value in pose[:3]) > 2.0 or max(abs(value) for value in pose[3:]) > 2 * math.pi:
        raise RobotConfigInputError(f"{label}超出保护范围。")
    return pose


def _serialize_reference_anchors(
    anchors: Mapping[str, Mapping[str, list[float] | None]],
) -> dict[str, dict[str, list[float] | str]]:
    return {
        scene: {color: anchors[scene][color] if anchors[scene][color] is not None else "UNSET" for color in COLORS}
        for scene in REFERENCE_LABELS
    }


def convert_tcp_display(values: Sequence[str], *, from_units: str, to_units: str) -> list[str]:
    parsed = _six_optional(values)
    if parsed is None or from_units == to_units:
        return list(values)
    if from_units == "m_rad" and to_units == "mm_deg":
        converted = [value * 1000.0 for value in parsed[:3]] + [math.degrees(value) for value in parsed[3:]]
    elif from_units == "mm_deg" and to_units == "m_rad":
        converted = [value / 1000.0 for value in parsed[:3]] + [math.radians(value) for value in parsed[3:]]
    else:
        raise RobotConfigInputError("未知 TCP显示单位。")
    return [_format(value) for value in converted]


def convert_joint_display(values: Sequence[str], *, from_units: str, to_units: str) -> list[str]:
    parsed = _six_optional(values)
    if parsed is None or from_units == to_units:
        return list(values)
    if from_units == "rad" and to_units == "deg":
        converted = [math.degrees(value) for value in parsed]
    elif from_units == "deg" and to_units == "rad":
        converted = [math.radians(value) for value in parsed]
    else:
        raise RobotConfigInputError("未知关节显示单位。")
    return [_format(value) for value in converted]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobotConfigInputError(f"配置文件无法读取：{path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise RobotConfigInputError(f"配置顶层不是对象：{path.name}")
    return value


def _optional_six(value: Any) -> list[float] | None:
    if value == "UNSET" or value is None:
        return None
    try:
        return _six_finite(value, "配置六维值")
    except RobotConfigInputError:
        return None


def _six_finite(values: Sequence[Any], label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != 6:
        raise RobotConfigInputError(f"{label}必须包含六个数值。")
    result: list[float] = []
    for index, value in enumerate(values, start=1):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RobotConfigInputError(f"{label}第{index}项不是数值。") from exc
        if not math.isfinite(number):
            raise RobotConfigInputError(f"{label}第{index}项不是有限数值。")
        result.append(number)
    return result


def _six_optional(values: Sequence[str]) -> list[float] | None:
    if len(values) != 6 or any(not str(value).strip() for value in values):
        return None
    try:
        return _six_finite(values, "显示值")
    except RobotConfigInputError:
        return None


def _format(value: float) -> str:
    return f"{value:.12g}"


def _write_temp(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _replace_two_json(first_path: Path, first: dict[str, Any], second_path: Path, second: dict[str, Any]) -> None:
    first_original = first_path.read_bytes()
    first_temp = _write_temp(first_path, first)
    second_temp = _write_temp(second_path, second)
    try:
        os.replace(first_temp, first_path)
        try:
            os.replace(second_temp, second_path)
        except Exception:
            rollback = first_path.with_name(f".{first_path.name}.rollback.tmp")
            rollback.write_bytes(first_original)
            os.replace(rollback, first_path)
            raise
    finally:
        first_temp.unlink(missing_ok=True)
        second_temp.unlink(missing_ok=True)
