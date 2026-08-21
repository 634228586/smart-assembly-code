from __future__ import annotations

import queue
import math
import threading
import time
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


class CountdownInput:
    """Interruptible automatic replacement for the wake word and task-card command."""

    def __init__(
        self,
        *,
        wake_phrase: str = "小具同学",
        command_phrase: str = "请开始识别任务卡",
        wakeup_delay_s: float = 5.0,
        command_delay_s: float = 5.0,
        next_command_delay_s: float | None = None,
    ) -> None:
        normalized_next_delay = command_delay_s if next_command_delay_s is None else next_command_delay_s
        if wakeup_delay_s < 0 or command_delay_s < 0 or normalized_next_delay < 0:
            raise ValueError("倒计时秒数不能为负数。")
        self.wake_phrase = wake_phrase
        self.command_phrase = command_phrase
        self.wakeup_delay_s = float(wakeup_delay_s)
        self.command_delay_s = float(command_delay_s)
        self.next_command_delay_s = float(normalized_next_delay)

    @staticmethod
    def _countdown(
        delay_s: float,
        *,
        stop_event: threading.Event,
        progress: Callable[[dict[str, Any]], None],
        phase: str,
        action: str,
    ) -> None:
        deadline = time.monotonic() + delay_s
        last_remaining: int | None = None
        while True:
            if stop_event.is_set():
                raise ManualTextInputStopped("收到人工停止请求，已取消倒计时自动指令。")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            remaining = max(1, int(math.ceil(remaining_s)))
            if remaining != last_remaining:
                progress({
                    "phase": phase,
                    "message": f"倒计时：{remaining}秒后{action}。",
                    "remaining_s": remaining,
                })
                last_remaining = remaining
            if stop_event.wait(min(0.1, remaining_s)):
                raise ManualTextInputStopped("收到人工停止请求，已取消倒计时自动指令。")

    def listen(
        self,
        wakeup_required: bool,
        *,
        stop_event: threading.Event,
        progress: Callable[[dict[str, Any]], None],
        on_wakeup: Callable[[], None],
    ) -> str:
        if wakeup_required:
            self._countdown(
                self.wakeup_delay_s,
                stop_event=stop_event,
                progress=progress,
                phase="countdown_waiting_wakeup",
                action=f"自动发送“{self.wake_phrase}”",
            )
            progress({
                "phase": "countdown_wakeup_sent",
                "message": f"倒计时结束，已自动发送：{self.wake_phrase}",
                "text": self.wake_phrase,
            })
            on_wakeup()

        command_delay_s = self.command_delay_s if wakeup_required else self.next_command_delay_s
        self._countdown(
            command_delay_s,
            stop_event=stop_event,
            progress=progress,
            phase="countdown_waiting_command",
            action=f"自动发送“{self.command_phrase}”",
        )
        progress({
            "phase": "countdown_command_sent",
            "message": f"倒计时结束，已自动发送：{self.command_phrase}",
            "text": self.command_phrase,
        })
        return self.command_phrase
