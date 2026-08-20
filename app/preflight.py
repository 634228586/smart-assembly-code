from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError, endpoints, load_all, walk_unset
from .nine_point import validate_direction_results
from .paths import PACKAGE_ROOT, REAL_CALIBRATION_DIR, REAL_CONFIG_DIR


@dataclass(frozen=True)
class Check:
    category: str
    item: str
    actual: str
    expected: str
    status: str
    action: str
    critical: bool = True


def _check(category: str, item: str, ok: bool, actual: Any, expected: str, action: str, *, critical: bool = True) -> Check:
    return Check(category, item, str(actual), expected, "PASS" if ok else ("FAIL" if critical else "WARN"), action, critical)


def _calibration_check(scene: str, robot_serial: Any, tcp_name: Any, scene_settings: dict[str, Any], profile: dict[str, Any]) -> Check:
    directory = REAL_CALIBRATION_DIR / scene
    files = sorted(directory.glob("*.json"))
    if len(files) != 1:
        return _check("真实标定", scene, False, f"{len(files)} files", "恰好 1 个已批准 runtime JSON", "完成真实九点标定并批准。")
    try:
        import json
        data = json.loads(files[0].read_text(encoding="utf-8"))
        required = {
            "schema_version": 1,
            "scene": scene,
            "data_origin": "camera_vision",
            "usable_for_real_robot": True,
            "robot_serial": robot_serial,
            "active_tcp": tcp_name,
            "photo_point": f"{scene}_photo",
            "step_x_mm": scene_settings.get("step_x_mm"),
            "step_y_mm": scene_settings.get("step_y_mm"),
            "image_width": profile.get("roi", {}).get("width"),
            "image_height": profile.get("roi", {}).get("height"),
        }
        ok = all(data.get(key) == value for key, value in required.items()) and data.get("approved") is True
        references = data.get("reference_detections")
        ok = ok and isinstance(references, dict) and set(references) == {"红", "橙", "黄", "绿", "蓝", "紫"}
        if ok:
            validate_direction_results(data.get("direction_validation", {}))
    except Exception as exc:
        return _check("真实标定", scene, False, type(exc).__name__, "可读取且身份一致", "修复或重新生成标定文件。")
    return _check("真实标定", scene, ok, files[0].name, "真实来源、身份一致且已批准", "重新标定或批准。")


def run_static_preflight() -> list[Check]:
    checks: list[Check] = []
    try:
        import json
        release = json.loads((PACKAGE_ROOT / "VERSION.json").read_text(encoding="utf-8"))
        release_ready = release.get("status") == "hardware_validated" and release.get("robot_motion_authorized") is True
        release_actual = {
            "status": release.get("status"),
            "robot_motion_authorized": release.get("robot_motion_authorized"),
        }
    except (OSError, ValueError, TypeError) as exc:
        release_ready = False
        release_actual = type(exc).__name__
    checks.append(_check(
        "发布状态", "真实硬件验收放行", release_ready, release_actual,
        "status=hardware_validated 且 robot_motion_authorized=true",
        "VERSION.json 发布状态仅作提示，不再阻止建立比赛会话授权。",
        critical=False,
    ))
    try:
        configs = load_all()
        checks.append(_check("配置", "完整配置集", True, "7 files", "7 files", ""))
    except ConfigurationError as exc:
        return [_check("配置", "完整配置集", False, exc, "全部 schema 1 配置可读取", "修复 config/real。")]

    # 只显示具有独立处理意义的基础配置；robot/camera由下方专项检查覆盖，
    # baseline已按现场决定取消，避免同一缺项重复计数。
    for name in ("endpoints", "motion", "suction_io", "competition"):
        value = configs[name]
        missing = list(walk_unset(value))
        checks.append(_check("配置", name, not missing, "完整" if not missing else ", ".join(missing), "无 UNSET", "在现场配置页补齐并核验。"))

    try:
        loaded_endpoints = endpoints(configs["endpoints"])
        checks.append(_check("端口", "角色与监听冲突", True, ", ".join(f"{k}={v.host}:{v.port}" for k, v in loaded_endpoints.items()), "角色唯一且端口合法", ""))
    except ConfigurationError as exc:
        checks.append(_check("端口", "角色与监听冲突", False, exc, "角色唯一且端口合法", "修复 endpoints.json。"))

    for module in ("PySide6", "pyaubo_sdk", "cv2", "yaml", "openai"):
        checks.append(_check("软件环境", module, importlib.util.find_spec(module) is not None, "已安装" if importlib.util.find_spec(module) else "缺失", "可导入", "安装 requirements-lock.txt 中的锁定版本。"))

    mvs_candidates = (
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll"),
    )
    found_mvs = next((path for path in mvs_candidates if path.is_file()), None)
    checks.append(_check("软件环境", "MVS Runtime", found_mvs is not None, found_mvs or "未发现", "官方 MvCameraControl.dll", "安装与相机匹配的官方 MVS SDK/Runtime。"))
    wrapper = PACKAGE_ROOT / "vision" / "vendor" / "mvs" / "MvCameraControl_class.py"
    checks.append(_check("软件环境", "包内 MVS Python wrapper", wrapper.is_file(), wrapper if wrapper.is_file() else "缺失", "厂商 Python wrapper随正式包携带", "从当前已安装的官方 MVS SDK重新复制 wrapper。"))
    real_vision_service = PACKAGE_ROOT / "vision" / "real_mvs_service.py"
    checks.append(_check("软件环境", "包内真实 MVS 服务", real_vision_service.is_file(), "已实现" if real_vision_service.is_file() else "未实现", "包内可启动且只允许真实 MVS来源", "恢复正式服务文件。"))

    for variable in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"):
        checks.append(_check("云服务", variable, bool(os.environ.get(variable)), "存在" if os.environ.get(variable) else "缺失", "环境变量存在", "设置当前 Windows 用户环境变量；不要写进比赛包。"))

    robot = configs["robot"]
    camera = configs["camera"]
    motion = configs["motion"]
    io = configs["suction_io"]
    aperture = io.get("aperture", {})
    aperture_configured = (
        aperture.get("output_type") == "standard_digital_output"
        and aperture.get("output_index") == 0
        and aperture.get("on_level") is True
        and aperture.get("off_level") is False
    )
    checks.append(_check("光圈", "DO0光圈配置", aperture_configured, aperture, "DO0：打开=true，关闭=false", "按已确认接线配置 DO0。"))
    profile_approval = {name: value.get("approved") for name, value in camera.get("profiles", {}).items() if isinstance(value, dict)}
    checks.append(_check("相机", "三套采集参数批准", set(profile_approval) == {"task_card", "blocks", "trays"} and all(profile_approval.values()), profile_approval, "三套 profile均已由现场操作者确认批准", "在相机与视觉页勾选真实数据确认并保存/人工批准；无需预先连接视觉服务回读。"))
    detector_configured = {}
    for name in ("blocks", "trays"):
        detector = camera.get("profiles", {}).get(name, {}).get("detector", {})
        required_detector = {
            key: detector.get(key) if isinstance(detector, dict) else None
            for key in ("roi", "confidence_min", "min_area_px", "max_area_px", "hsv_ranges")
        }
        detector_configured[name] = isinstance(detector, dict) and not list(walk_unset(required_detector))
    checks.append(_check(
        "视觉", "两套颜色参数完整", all(detector_configured.values()), detector_configured,
        "blocks/trays均包含ROI、面积、置信度和六色HSV", "采集当前现场实拍图，离线分析后直接更新 camera.json。",
        critical=False,
    ))
    checks.append(_check(
        "运动", "直接 MoveJoint策略",
        motion.get("direct_route_approval_required") is False,
        "不要求路线验收；程序不提供碰撞规划",
        "比赛启动前由操作者一次确认全部直接路径无人员和障碍物",
        "保持现场清空并确认起点；每次运动仍检查机器人状态和活动 TCP。",
    ))

    robot_serial = robot["identity"].get("binding_id")
    tcp_name = robot["active_tcp"].get("name")
    checks.append(_calibration_check("blocks", robot_serial, tcp_name, motion.get("nine_point", {}).get("blocks", {}), camera.get("profiles", {}).get("blocks", {})))
    checks.append(_calibration_check("trays", robot_serial, tcp_name, motion.get("nine_point", {}).get("trays", {}), camera.get("profiles", {}).get("trays", {})))

    checks.append(_check("完整性", "包路径独立", PACKAGE_ROOT.name == "装配赛正式代码", PACKAGE_ROOT, "装配赛正式代码", "从正式目录启动。", critical=False))
    return checks


def competition_ready(checks: list[Check]) -> bool:
    return bool(checks) and all(item.status == "PASS" for item in checks if item.critical)
