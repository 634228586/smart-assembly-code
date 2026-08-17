from __future__ import annotations

import queue
import threading
from typing import Any, Callable


class ManualTextInputStopped(RuntimeError):
    pass


class ManualTextInput:
    """Thread-safe replacement for ASR input; it never executes hardware itself."""

    def __init__(self, wake_phrase: str = "小具同学") -> None:
        self.wake_phrase = wake_phrase
        self._commands: queue.Queue[str] = queue.Queue()

    @staticmethod
    def _normalized(text: str) -> str:
        return "".join(text.split())

    def submit(self, text: str) -> str:
        value = text.strip()
        if not value:
            raise ValueError("文字指令不能为空。")
        self._commands.put(value)
        return value

    def _next(self, stop_event: threading.Event) -> str:
        while not stop_event.is_set():
            try:
                return self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
        raise ManualTextInputStopped("收到人工停止请求，已停止等待文字指令。")

    def listen(
        self,
        wakeup_required: bool,
        *,
        stop_event: threading.Event,
        progress: Callable[[dict[str, Any]], None],
        on_wakeup: Callable[[], None],
    ) -> str:
        if wakeup_required:
            progress({"phase": "manual_text_waiting_wakeup", "message": f"文字模式：请输入“{self.wake_phrase}”唤醒。"})
            while True:
                text = self._next(stop_event)
                progress({"phase": "manual_text_received", "message": f"收到文字：{text}"})
                if self._normalized(text) == self._normalized(self.wake_phrase):
                    on_wakeup()
                    break
                progress({"phase": "manual_text_ignored", "message": f"尚未唤醒，已忽略：{text}"})

        progress({"phase": "manual_text_waiting_command", "message": "文字模式：请输入任务指令。"})
        instruction = self._next(stop_event)
        progress({"phase": "manual_text_received", "message": f"收到文字：{instruction}"})
        return instruction
