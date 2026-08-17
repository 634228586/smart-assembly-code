from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.contracts import CapturedFrame


class RecognitionError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None, model: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.model = model


SYSTEM_PROMPT = """你是装配赛任务卡识别器。只输出一个 JSON 对象，不要 Markdown。
识别成功必须输出 success=true。任务卡一输出 task_type=task_1、scene_description 和 confidence。
任务卡二输出 task_type=task_2、confidence，以及恰好六项 sequence；每项只含 order、block_color、tray_color。
颜色只能是红、橙、黄、绿、蓝、紫；任务二的方块颜色和托盘颜色都必须各自完整且不重复。
无法可靠判断时输出 success=false、task_type=unknown、error_code 和 message。"""

MODEL_IMAGE_MAX_EDGE_PX = 1600
MODEL_IMAGE_MAX_BYTES = 2_000_000


def _prepare_model_image(path: Path) -> Path:
    """Preserve the evidence image and create a bounded JPEG for the model API."""

    if not path.is_file():
        raise RecognitionError(f"本次任务卡图片不存在：{path}")
    try:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise RecognitionError(f"任务卡图片读取失败：{exc}") from exc
    if image is None or image.size == 0:
        raise RecognitionError("任务卡图片无法解码，不能发送给大模型。")

    height, width = image.shape[:2]
    longest = max(width, height)
    if longest > MODEL_IMAGE_MAX_EDGE_PX:
        scale = MODEL_IMAGE_MAX_EDGE_PX / float(longest)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    encoded: np.ndarray | None = None
    for quality in (85, 75, 65, 55):
        try:
            ok, candidate = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
        except cv2.error as exc:
            raise RecognitionError(f"任务卡模型图片压缩失败：{exc}") from exc
        if not ok:
            raise RecognitionError("任务卡模型图片压缩失败。")
        encoded = candidate
        if int(candidate.size) <= MODEL_IMAGE_MAX_BYTES:
            break
    if encoded is None or int(encoded.size) > MODEL_IMAGE_MAX_BYTES:
        raise RecognitionError(
            f"压缩后的任务卡图片仍超过{MODEL_IMAGE_MAX_BYTES / 1_000_000:.1f} MB，未发送给大模型。"
        )

    model_path = path.with_name(f"{path.stem}_qwen.jpg")
    try:
        model_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise RecognitionError(f"任务卡模型图片保存失败：{exc}") from exc
    return model_path


def _data_url(path: Path) -> str:
    model_path = _prepare_model_image(path)
    mime = mimetypes.guess_type(model_path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(model_path.read_bytes()).decode('ascii')}"


def _request_task_card(image_path: Path) -> tuple[dict[str, Any], str, str, str | None]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("DASHSCOPE_BASE_URL")
    model = os.environ.get("DASHSCOPE_MODEL")
    if not api_key or not base_url or not model:
        raise RecognitionError("DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL或DASHSCOPE_MODEL缺失。")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "识别这张本次实时拍摄的任务卡。"},
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ]},
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RecognitionError("Qwen没有返回文本。", model=model)
        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise RecognitionError(
                f"Qwen返回内容不是合法JSON：{exc}", raw_response=content, model=model,
            ) from exc
        if not isinstance(result, dict):
            raise RecognitionError("Qwen结果不是 JSON对象。", raw_response=content, model=model)
        response_id = getattr(response, "id", None)
        return result, content, model, str(response_id) if response_id is not None else None
    except RecognitionError:
        raise
    except Exception as exc:
        raise RecognitionError(f"Qwen调用或结果解析失败：{type(exc).__name__}: {exc}", model=model) from exc


def recognize_task_card(image_path: Path) -> dict[str, Any]:
    return _request_task_card(image_path)[0]


def recognize_task_card_with_diagnostics(image_path: Path) -> dict[str, Any]:
    """Use the formal model request once and expose non-secret diagnostic metadata."""

    started = time.perf_counter()
    result, raw_response, model, response_id = _request_task_card(image_path)
    return {
        "model_result": result,
        "raw_response": raw_response,
        "model": model,
        "provider_request_id": response_id,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def build_recognition_result(model_result: dict[str, Any], *, frame: CapturedFrame) -> dict[str, Any]:
    """把模型内容封装为总控严格协议，不允许模型伪造相机或请求身份。"""

    if not isinstance(model_result, dict):
        raise RecognitionError("Qwen结果不是 JSON对象。")
    source = {
        "image_id": frame.image_path.stem,
        "path": str(frame.image_path.resolve()),
        "captured_at": frame.captured_at,
        "camera_serial": frame.camera_serial,
        "capture_request_id": frame.request_id,
    }
    raw_text = json.dumps(model_result, ensure_ascii=False, allow_nan=False)
    common = {
        "schema_version": 1,
        "type": "recognition_result",
        "request_id": frame.request_id,
        "task_type": model_result.get("task_type", "unknown"),
        "success": model_result.get("success"),
        "recognized_at": datetime.now(timezone.utc).isoformat(),
        "raw_text": raw_text,
        "source_image": source,
    }
    if model_result.get("success") is not True:
        return {
            **common,
            "success": False,
            "task_type": model_result.get("task_type", "unknown"),
            "error_code": str(model_result.get("error_code", "MODEL_UNCERTAIN")),
            "message": str(model_result.get("message", "识别失败")),
        }
    if model_result.get("task_type") == "task_1":
        return {**common, "confidence": model_result.get("confidence"), "scene_description": model_result.get("scene_description")}
    if model_result.get("task_type") == "task_2":
        return {**common, "confidence": model_result.get("confidence"), "sequence": model_result.get("sequence")}
    raise RecognitionError("Qwen成功结果缺少合法 task_type。")
