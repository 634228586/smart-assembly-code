from __future__ import annotations

import json
import http.client
import urllib.request
from typing import Any

from app.config import Endpoint
from app.protocols import ProtocolError, normalize_speech_health


class SpeechError(RuntimeError):
    pass


def _request(
    endpoint: Endpoint,
    method: str,
    route: str,
    payload: dict[str, Any] | None,
    timeout: float,
    *,
    retry_remote_disconnect: bool = False,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts = 2 if retry_remote_disconnect else 1
    for attempt in range(attempts):
        request = urllib.request.Request(
            f"http://{endpoint.host}:{endpoint.port}{route}", data=data, method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read(65536).decode("utf-8"))
            break
        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as exc:
            if attempt + 1 < attempts:
                continue
            raise SpeechError(f"语音服务请求失败：{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise SpeechError(f"语音服务请求失败：{type(exc).__name__}: {exc}") from exc
    if not isinstance(result, dict):
        raise SpeechError("语音服务响应不是 JSON对象。")
    return result


def health(endpoint: Endpoint) -> dict[str, Any]:
    result = _request(endpoint, "GET", endpoint.routes["health"], None, 5.0)
    try:
        return normalize_speech_health(result, expected_service=endpoint.expected_service)
    except ProtocolError as exc:
        raise SpeechError(str(exc)) from exc


def listen(endpoint: Endpoint, *, wakeup_required: bool, timeout_s: float) -> str:
    result = _request(endpoint, "POST", endpoint.routes["asr"], {
        "wakeup_required": wakeup_required,
        "prewoken": not wakeup_required,
        "wakeup_timeout_s": timeout_s,
        "wakeup_feedback_text": "我已就绪，请下达指令" if wakeup_required else "",
    }, timeout_s + 30.0)
    if result.get("ok") is not True:
        raise SpeechError(f"ASR失败 [{result.get('error_code', 'ASR_ERROR')}]：{result.get('message', '')}")
    digest = result.get("result_digest") if isinstance(result.get("result_digest"), dict) else {}
    instruction = result.get("instruction") or digest.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SpeechError("ASR没有返回有效指令。")
    return instruction.strip()


def speak(
    endpoint: Endpoint,
    text: str,
    *,
    timeout_s: float = 90.0,
    retry_remote_disconnect: bool = True,
) -> None:
    if timeout_s <= 0:
        raise ValueError("TTS超时秒数必须大于0。")
    result = _request(
        endpoint, "POST", endpoint.routes["tts"], {"text": text}, float(timeout_s),
        retry_remote_disconnect=retry_remote_disconnect,
    )
    if result.get("ok") is not True:
        raise SpeechError(f"TTS失败 [{result.get('error_code', 'TTS_ERROR')}]：{result.get('message', '')}")
    digest = result.get("result_digest")
    if not isinstance(digest, dict):
        raise SpeechError("TTS响应缺少真实播报摘要。")
    if digest.get("tts_skipped") is True:
        raise SpeechError(f"TTS实际未播报 [{digest.get('skip_code', 'TTS_SKIPPED')}]：{result.get('message', '')}")
    if digest.get("spoken_text") != text:
        raise SpeechError("TTS回报的实际播报文字与请求不一致。")
