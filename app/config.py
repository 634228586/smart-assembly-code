from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .paths import REAL_CONFIG_DIR


UNSET = "UNSET"
REQUIRED_CONFIG_FILES = (
    "endpoints.json",
    "robot.json",
    "camera.json",
    "motion.json",
    "suction_io.json",
    "competition.json",
    "baseline.json",
)


class ConfigurationError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"缺少配置文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"配置不是有效 JSON：{path}：{exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"配置顶层必须是对象：{path}")
    if value.get("schema_version") != 1:
        raise ConfigurationError(f"不支持的配置版本：{path}")
    return value


def load_all() -> dict[str, dict[str, Any]]:
    return {Path(name).stem: load_json(REAL_CONFIG_DIR / name) for name in REQUIRED_CONFIG_FILES}


def walk_unset(value: Any, prefix: str = "") -> Iterable[str]:
    if value == UNSET:
        yield prefix or "<root>"
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_unset(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_unset(child, f"{prefix}[{index}]")


def canonical_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix().lower()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class Endpoint:
    role: str
    host: str
    port: int
    direction: str
    expected_service: str
    routes: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


def _routes(role: str, raw: dict[str, Any]) -> Mapping[str, str]:
    value = raw.get("routes", {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"端点 {role}.routes 必须是对象。")
    if role == "speech_service" and set(value) != {"health", "asr", "tts"}:
        raise ConfigurationError("speech_service.routes 必须恰好包含 health、asr、tts。")
    result: dict[str, str] = {}
    for name, route in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(route, str)
            or route != route.strip()
            or not route.startswith("/")
            or route.startswith("//")
            or any(character in route for character in "?#\r\n")
        ):
            raise ConfigurationError(f"端点 {role}.routes.{name} 必须是不含查询或片段的站内绝对路径。")
        result[name] = route
    return MappingProxyType(result)


def endpoints(config: dict[str, Any]) -> dict[str, Endpoint]:
    result: dict[str, Endpoint] = {}
    for role in ("robot_rpc", "vision_service", "qt_command_listener", "speech_service"):
        raw = config.get(role)
        if not isinstance(raw, dict):
            raise ConfigurationError(f"端点 {role} 缺失。")
        host, port = raw.get("host"), raw.get("port")
        if host == UNSET or port == UNSET:
            continue
        if not isinstance(host, str) or not host.strip():
            raise ConfigurationError(f"端点 {role}.host 无效。")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ConfigurationError(f"端点 {role}.port 必须为 1..65535。")
        direction = raw.get("direction")
        service = raw.get("expected_service")
        if direction not in {"listen", "outbound"} or not isinstance(service, str) or not service:
            raise ConfigurationError(f"端点 {role} 的方向或服务身份无效。")
        result[role] = Endpoint(role, host.strip(), port, direction, service, _routes(role, raw))

    listeners: set[tuple[str, int]] = set()
    for endpoint in result.values():
        if endpoint.direction != "listen":
            continue
        key = (endpoint.host, endpoint.port)
        if key in listeners:
            raise ConfigurationError(f"本机监听端点重复：{endpoint.host}:{endpoint.port}")
        listeners.add(key)
    return result
