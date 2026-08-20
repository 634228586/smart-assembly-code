from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from voice import speech_client
from voice.qwen_recognizer import recognize_task_card

from .config import endpoints, load_all
from .coordinator import CompetitionCoordinator
from .integrity import current_runtime_fingerprint
from .manual_text_input import CountdownInput, ManualTextInput
from .paths import REAL_CALIBRATION_DIR, REAL_CONFIG_DIR, SESSION_DIR
from .real_ports import ConfiguredRobotPort
from .robot_gateway import AuboRealGateway
from .session import CompetitionSession
from .vision_client import RealVisionClient


class RuntimeBuildError(RuntimeError):
    pass


def _calibration_id(scene: str) -> str:
    files = sorted((REAL_CALIBRATION_DIR / scene).glob("*.json"))
    if len(files) != 1:
        raise RuntimeBuildError(f"{scene} 必须恰好存在一个已批准的真实标定文件。")
    try:
        value = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeBuildError(f"{scene} 标定文件无法读取：{exc}") from exc
    calibration_id = value.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip() or value.get("approved") is not True:
        raise RuntimeBuildError(f"{scene} 标定缺少已批准的 calibration_id。")
    return calibration_id.strip()


@dataclass
class RealCompetitionRuntime:
    coordinator: CompetitionCoordinator
    gateway: AuboRealGateway

    def close(self) -> None:
        self.gateway.disconnect()


def build_real_runtime(
    *,
    session: CompetitionSession,
    stop_event: threading.Event,
    progress: Callable[[dict[str, Any]], None],
    manual_text_input: ManualTextInput | None = None,
    input_mode: str = "voice",
) -> RealCompetitionRuntime:
    """仅由用户点击正式启动后在 Robot Worker 线程调用。"""

    configs = load_all()
    if input_mode not in {"voice", "text", "countdown"}:
        raise RuntimeBuildError(f"不支持的输入模式：{input_mode}")
    if input_mode == "text" and manual_text_input is None:
        raise RuntimeBuildError("文字控制模式缺少文字输入通道。")
    configured_endpoints = endpoints(configs["endpoints"])
    for role in ("robot_rpc", "vision_service", "speech_service"):
        if role not in configured_endpoints:
            raise RuntimeBuildError(f"端点 {role} 尚未配置。")
    fingerprint = current_runtime_fingerprint()

    session_id = datetime.now().strftime("competition-%Y%m%d-%H%M%S-%f")
    gateway = AuboRealGateway(configs["endpoints"]["robot_rpc"], configs["robot"])
    try:
        snapshot = gateway.connect_readonly()
        progress({"phase": "robot_identity", "message": f"真实机器人只读身份通过：{snapshot.robot_name}"})
        calibration_ids = {scene: _calibration_id(scene) for scene in ("blocks", "trays")}
        vision = RealVisionClient(
            configured_endpoints["vision_service"],
            active_tcp=configs["robot"]["active_tcp"]["name"],
            calibration_ids=calibration_ids,
            fresh_frame_max_age_ms=int(configs["camera"]["fresh_frame_max_age_ms"]),
            visual_result_callback=lambda payload: progress({**payload, "phase": "visual_result"}),
        )
        vision.health()
        progress({"phase": "vision_identity", "message": "真实 MVS视觉服务已就绪。"})
        speech_endpoint = configured_endpoints["speech_service"]
        trigger = configs["competition"]["task_trigger"]
        command_input: ManualTextInput | CountdownInput | None = manual_text_input
        if input_mode == "countdown":
            countdown = configs["competition"].get("countdown_control", {})
            try:
                wakeup_delay_s = float(countdown["wakeup_delay_s"])
                command_delay_s = float(countdown["command_delay_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeBuildError("competition.json 缺少有效的倒计时控制秒数。") from exc
            command_input = CountdownInput(
                wake_phrase=str(trigger["wake_word"]),
                command_phrase=str(trigger["phrase"]),
                wakeup_delay_s=wakeup_delay_s,
                command_delay_s=command_delay_s,
            )

        if command_input is None:
            speech_client.health(speech_endpoint)
            progress({"phase": "speech_identity", "message": "AI语音盒子身份与 ASR/TTS能力通过。"})
        elif input_mode == "countdown":
            progress({
                "phase": "countdown_mode",
                "message": "倒计时控制已启用：首次自动唤醒并发送识别指令，后续自动发送识别指令；跳过ASR健康门禁。",
            })
        else:
            progress({
                "phase": "manual_text_mode",
                "message": "文字控制已启用：跳过 ASR 健康门禁；TTS仍会尽力播报。",
            })

        robot = ConfiguredRobotPort(
            gateway,
            session_id=session_id,
            config_fingerprint=fingerprint,
            motion=configs["motion"],
            suction_io=configs["suction_io"],
            stop_event=stop_event,
            fingerprint_provider=current_runtime_fingerprint,
            on_command=lambda command: progress({"phase": "robot_command", "message": command}),
        )
        def speak_best_effort(text: str) -> None:
            try:
                speech_client.speak(speech_endpoint, text)
            except Exception as exc:
                progress({"phase": "tts_warning", "message": f"语音播报失败，不阻止当前控制流程：{exc}"})

        def text_wakeup() -> None:
            reply = "我已就绪，请下达指令"
            phase = "countdown_reply" if input_mode == "countdown" else "manual_text_reply"
            progress({"phase": phase, "message": reply})
            speak_best_effort(reply)

        listener = (
            (lambda wakeup: speech_client.listen(speech_endpoint, wakeup_required=wakeup, timeout_s=30.0))
            if command_input is None
            else (
                lambda wakeup: command_input.listen(
                    wakeup,
                    stop_event=stop_event,
                    progress=progress,
                    on_wakeup=text_wakeup,
                )
            )
        )
        coordinator = CompetitionCoordinator(
            session=session,
            robot=robot,
            vision=vision,
            recognizer=recognize_task_card,
            listener=listener,
            speaker=lambda text: speech_client.speak(speech_endpoint, text),
            session_id=session_id,
            session_dir=SESSION_DIR / session_id,
            config_fingerprint=fingerprint,
            points=configs["motion"]["points"],
            reference_anchors=configs["motion"]["reference_anchors"],
            phrase=trigger["phrase"],
            progress=progress,
            stop_event=stop_event,
        )
        return RealCompetitionRuntime(coordinator, gateway)
    except Exception:
        gateway.disconnect()
        raise
