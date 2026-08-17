from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QObject, Signal, Slot

from voice import speech_client

from .config import endpoints, load_all


VoiceAction = Literal["health", "listen", "speak"]


class VoiceInteractionWorker(QObject):
    """Run one explicit AI-box test without touching the robot or vision service."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        action: VoiceAction,
        *,
        wakeup_required: bool = False,
        timeout_s: float = 30.0,
        text: str = "",
    ) -> None:
        super().__init__()
        self.action = action
        self.wakeup_required = wakeup_required
        self.timeout_s = timeout_s
        self.text = text

    @Slot()
    def run(self) -> None:
        try:
            configured = endpoints(load_all()["endpoints"])
            endpoint = configured.get("speech_service")
            if endpoint is None:
                raise RuntimeError("config/real/endpoints.json 中未配置语音盒子。")

            if self.action == "health":
                health = speech_client.health(endpoint)
                result = {"action": "health", "health": health}
            elif self.action == "listen":
                # Verify the configured service identity before opening a real ASR session.
                speech_client.health(endpoint)
                recognized = speech_client.listen(
                    endpoint,
                    wakeup_required=self.wakeup_required,
                    timeout_s=self.timeout_s,
                )
                result = {"action": "listen", "recognized_text": recognized}
            elif self.action == "speak":
                value = self.text.strip()
                if not value:
                    raise ValueError("播报文字不能为空。")
                speech_client.health(endpoint)
                speech_client.speak(endpoint, value)
                result = {"action": "speak", "spoken_text": value}
            else:
                raise ValueError(f"不支持的语音测试操作：{self.action}")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)
