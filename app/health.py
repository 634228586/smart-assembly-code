from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Endpoint
from .protocols import ProtocolError, normalize_speech_health, validate_service_identity


@dataclass(frozen=True)
class ServiceHealth:
    role: str
    host: str
    port: int
    identity: str
    protocol_version: int
    details: dict


def probe_json_line_service(endpoint: Endpoint, *, timeout_s: float = 2.0) -> ServiceHealth:
    """验证 JSON 行服务身份；不允许把“TCP 可连接”当作健康。"""

    request = {"type": "health_request", "protocol_version": 1, "expected_service": endpoint.expected_service}
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout_s) as connection:
            connection.settimeout(timeout_s)
            connection.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            data = bytearray()
            while not data.endswith(b"\n"):
                chunk = connection.recv(4096)
                if not chunk:
                    raise ProtocolError("服务在返回完整健康响应前关闭连接。")
                data.extend(chunk)
                if len(data) > 65536:
                    raise ProtocolError("健康响应超过 64 KiB。")
        payload = json.loads(data.decode("utf-8"))
        validate_service_identity(payload, expected_service=endpoint.expected_service)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{endpoint.role} 健康检查失败：{exc}") from exc
    return ServiceHealth(endpoint.role, endpoint.host, endpoint.port, payload["service"], payload["protocol_version"], payload)


def probe_speech_http(endpoint: Endpoint, *, timeout_s: float = 3.0) -> ServiceHealth:
    url = f"http://{endpoint.host}:{endpoint.port}{endpoint.routes['health']}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read(65536).decode("utf-8"))
        payload = normalize_speech_health(payload, expected_service=endpoint.expected_service)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"speech_service 健康检查失败：{exc}") from exc
    return ServiceHealth(endpoint.role, endpoint.host, endpoint.port, payload["service"], payload["protocol_version"], payload)
