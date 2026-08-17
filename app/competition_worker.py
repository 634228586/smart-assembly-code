from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from .runtime import RealCompetitionRuntime, build_real_runtime
from .manual_text_input import ManualTextInput
from .session import CompetitionSession


class CompetitionWorker(QObject):
    progress = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, session: CompetitionSession, *, input_mode: str = "voice") -> None:
        super().__init__()
        if input_mode not in {"voice", "text"}:
            raise ValueError(f"不支持的输入模式：{input_mode}")
        self.session = session
        self.input_mode = input_mode
        self.stop_event = threading.Event()
        self.manual_text_input = ManualTextInput() if input_mode == "text" else None
        self.runtime: RealCompetitionRuntime | None = None

    @Slot()
    def run(self) -> None:
        try:
            self.runtime = build_real_runtime(
                session=self.session,
                stop_event=self.stop_event,
                progress=self.progress.emit,
                manual_text_input=self.manual_text_input,
            )
            self.runtime.coordinator.run()
        except Exception as exc:
            if self.session.state.value not in {"stopped", "complete"}:
                self.session.revoke(f"正式流程失败：{exc}")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
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
