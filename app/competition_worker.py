from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .runtime import RealCompetitionRuntime, build_real_runtime
from .manual_text_input import ManualTextInput
from .paths import LOG_DIR
from .session import CompetitionSession


class CompetitionWorker(QObject):
    progress = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, session: CompetitionSession, *, input_mode: str = "voice") -> None:
        super().__init__()
        if input_mode not in {"voice", "text", "countdown"}:
            raise ValueError(f"不支持的输入模式：{input_mode}")
        self.session = session
        self.input_mode = input_mode
        self.stop_event = threading.Event()
        self.manual_text_input = ManualTextInput() if input_mode == "text" else None
        self.runtime: RealCompetitionRuntime | None = None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.event_log_path = LOG_DIR / "controller" / f"competition-{stamp}.jsonl"

    def _record_progress(self, event: object) -> None:
        value: dict[str, Any] = dict(event) if isinstance(event, dict) else {
            "phase": "event",
            "message": str(event),
        }
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "input_mode": self.input_mode,
            **value,
        }
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.progress.emit({
                "phase": "log_write_warning",
                "message": f"持久化通信日志写入失败：{exc}",
            })
        self.progress.emit(value)

    def _record_terminal_event(self, phase: str, message: str) -> None:
        self._record_progress({"phase": phase, "message": message})

    @Slot()
    def run(self) -> None:
        try:
            self._record_progress({
                "phase": "communication_log_started",
                "message": f"本场通信日志：{self.event_log_path}",
                "log_path": str(self.event_log_path),
            })
            self.runtime = build_real_runtime(
                session=self.session,
                stop_event=self.stop_event,
                progress=self._record_progress,
                manual_text_input=self.manual_text_input,
                input_mode=self.input_mode,
            )
            self.runtime.coordinator.run()
        except Exception as exc:
            if self.session.state.value not in {"stopped", "complete"}:
                self.session.revoke(f"正式流程失败：{exc}")
            self._record_terminal_event("competition_failed", f"{type(exc).__name__}: {exc}")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self._record_terminal_event("competition_worker_finished", "正式双任务卡Worker执行完成。")
            self.finished.emit()
        finally:
            if self.runtime is not None:
                self.runtime.close()
                self.runtime = None

    def request_stop(self) -> None:
        """可由 GUI线程调用；只设置线程安全事件，不直接调用 SDK或跨线程改状态。"""

        self.stop_event.set()

    def submit_text(self, text: str) -> str:
        """May be called by the GUI thread; Queue.put is thread-safe."""

        if self.manual_text_input is None:
            raise RuntimeError("当前不是文字控制模式。")
        return self.manual_text_input.submit(text)
