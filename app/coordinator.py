from __future__ import annotations

import secrets
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

from vision.contracts import CapturedFrame
from voice.qwen_recognizer import build_recognition_result
from voice.trigger import command_matches

from .assembly import AssemblyExecutor
from .execution_gate import TwoPhaseExecutionGate
from .protocols import validate_recognition_result
from .session import CompetitionSession, SessionState


class CoordinatorError(RuntimeError):
    pass


class TaskCardCamera(Protocol):
    def capture_task_card(self, *, request_id: str, session_id: str, session_dir: Path) -> CapturedFrame: ...


class CompetitionCoordinator:
    """正式双卡会话编排；必须在独占 Robot Worker 线程中运行。"""

    def __init__(
        self,
        *,
        session: CompetitionSession,
        robot: Any,
        vision: TaskCardCamera,
        recognizer: Callable[[Path], dict[str, Any]],
        listener: Callable[[bool], str],
        speaker: Callable[[str], None] | None,
        session_id: str,
        session_dir: Path,
        config_fingerprint: str,
        points: dict[str, list[float]],
        reference_anchors: dict[str, Any],
        phrase: str = "请开始识别任务卡",
        progress: Callable[[dict[str, Any]], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.session = session
        self.robot = robot
        self.vision = vision
        self.recognizer = recognizer
        self.listener = listener
        self.speaker = speaker
        self.session_id = session_id
        self.session_dir = session_dir.resolve()
        self.config_fingerprint = config_fingerprint
        self.points = points
        self.reference_anchors = reference_anchors
        self.phrase = phrase
        self._final_completion_spoken = False
        self.progress = progress or (lambda event: None)
        self.stop_event = stop_event or threading.Event()
        self.gate = TwoPhaseExecutionGate(session)
        self._executor: AssemblyExecutor | None = None

    def _emit(self, phase: str, message: str, **extra: Any) -> None:
        self.progress({"phase": phase, "message": message, **extra})

    def _speak_best_effort(self, text: str) -> None:
        if self.speaker is None:
            return
        try:
            self.speaker(text)
        except Exception as exc:
            self._emit("tts_warning", f"语音播报失败，不改变已验证的数据和硬件门控：{exc}")

    def request_stop(self) -> None:
        self.stop_event.set()
        if self._executor is not None:
            self._executor.request_stop()
        else:
            self.robot.request_stop()
        self.gate.cancel("人工停止")

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise CoordinatorError("收到人工停止请求，比赛授权已撤销。")

    def _recognize(self, frame: CapturedFrame) -> dict[str, Any]:
        try:
            model_result = self.recognizer(frame.image_path)
            formal = build_recognition_result(model_result, frame=frame)
            return validate_recognition_result(formal, session_dir=self.session_dir)
        except Exception as exc:
            self._emit("recognition_failed", f"识别失败：{exc}")
            self._speak_best_effort("识别失败")
            raise CoordinatorError(f"识别失败：{exc}") from exc

    def _process_result(self, result: dict[str, Any]) -> None:
        if result["success"] is not True:
            message = result.get("message", "识别失败")
            self._emit("recognition_failed", f"识别失败：{message}")
            self._speak_best_effort("识别失败")
            return
        if result["task_type"] == "task_1":
            outcome = self.session.accept_recognition(result)
            self._emit(outcome, str(result["scene_description"]))
            if outcome == "task_1_completed":
                if self.session.state == SessionState.COMPLETE:
                    self._speak_best_effort(
                        f"{result['scene_description']}。任务卡一识别完成，任务卡一和任务卡二均已完成。"
                    )
                    self._final_completion_spoken = True
                else:
                    self._speak_best_effort(
                        f"{result['scene_description']}。任务卡一识别完成，请更换下一张任务卡。"
                    )
            return

        prepared = self.gate.prepare(result, self.config_fingerprint)
        if prepared.get("type") != "ready_to_execute":
            self._emit(str(prepared.get("status", "task_2_rejected")), "任务卡二未进入执行。")
            return
        self._speak_best_effort("任务卡二识别完成，开始执行任务")
        recognition = self.gate.commit(
            request_id=result["request_id"],
            execute_token=prepared["execute_token"],
            config_fingerprint=self.config_fingerprint,
        )
        self._executor = AssemblyExecutor(
            self.robot,
            self.vision,
            session_id=self.session_id,
            points=self.points,
            reference_anchors=self.reference_anchors,
            progress=self.progress,
            cycle_completed=self.session.cycle_completed,
            stop_event=self.stop_event,
        )
        try:
            assembly_result = self._executor.run(recognition["sequence"])
        except Exception as exc:
            self.session.revoke("装夹执行失败", suction_state="可能仍为 ON" if self._executor.suction_may_be_on else "unknown")
            raise CoordinatorError(f"装夹失败，未继续后续动作：{exc}") from exc
        finally:
            self._executor = None
        if not assembly_result.returned_to_task_card or assembly_result.processed_cycles != 6:
            self.session.revoke("六组处理或返航未完成", suction_state="可能仍为 ON" if assembly_result.suction_may_be_on else "OFF")
            raise CoordinatorError("六组处理或返航未完成，不能确认任务二完成。")
        for _item in assembly_result.skipped:
            self.session.cycle_skipped()
        self.session.task_2_return_completed()
        self._emit(
            "task_2_completed",
            f"六组已处理：成功{assembly_result.completed_cycles}组，跳过{assembly_result.skipped_cycles}组；已返回任务卡拍照点。",
            completed_cycles=assembly_result.completed_cycles,
            skipped_cycles=assembly_result.skipped_cycles,
            skipped=list(assembly_result.skipped),
        )
        if self.session.state != SessionState.COMPLETE:
            self._speak_best_effort("任务卡二执行完成，我已返回任务卡拍照点，请下达指令。")

    def run(self) -> None:
        if self.session.state != SessionState.AUTHORIZED:
            raise CoordinatorError("比赛会话尚未授权。")
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._emit("task_card_photo_point", "移动到任务卡拍照点。")
        self.robot.move_joint(self.points["task_card_photo"])
        wakeup_required = True
        while self.session.state != SessionState.COMPLETE:
            self._check_stop()
            instruction = self.listener(wakeup_required)
            self._check_stop()
            if not command_matches(instruction, self.phrase):
                self._emit("instruction_ignored", f"未命中任务卡触发规则：{instruction}")
                continue
            self._emit(
                "command_matched",
                f"任务卡拍照指令已命中：{instruction}",
                instruction=instruction,
                configured_phrase=self.phrase,
            )
            wakeup_required = False
            request_id = f"{self.session_id}-{secrets.token_hex(8)}"
            self._emit("task_card_capture", "触发本次 MVS任务卡拍照。", request_id=request_id)
            try:
                frame = self.vision.capture_task_card(
                    request_id=request_id,
                    session_id=self.session_id,
                    session_dir=self.session_dir,
                )
                result = self._recognize(frame)
            except CoordinatorError:
                continue
            self._check_stop()
            self._process_result(result)
        if not self._final_completion_spoken:
            self._speak_best_effort("任务卡一和任务卡二均已完成。")
        self._emit("competition_complete", "本场双任务卡流程完成。")
