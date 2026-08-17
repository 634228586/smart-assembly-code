from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    IDLE = "idle"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    STOPPED = "stopped"
    COMPLETE = "complete"


class SessionError(RuntimeError):
    pass


@dataclass
class CompetitionSession:
    state: SessionState = SessionState.IDLE
    processed_task_types: set[str] = field(default_factory=set)
    current_request_id: str | None = None
    completed_cycles: int = 0
    skipped_cycles: int = 0
    suction_state: str = "unknown"
    authorization_revoked_reason: str | None = None

    def authorize(self, preflight_ready: bool) -> None:
        if not preflight_ready:
            raise SessionError("赛前检查未全部通过，不能建立比赛授权。")
        if self.state not in {SessionState.IDLE, SessionState.STOPPED}:
            raise SessionError("当前状态不能建立新授权。")
        self.state = SessionState.AUTHORIZED
        self.authorization_revoked_reason = None

    def accept_recognition(self, result: dict[str, Any]) -> str:
        if result.get("success") is not True:
            return "recognition_failed"
        task_type = str(result["task_type"])
        if task_type in self.processed_task_types:
            return "duplicate_ignored"
        if task_type == "task_1":
            self.processed_task_types.add(task_type)
            if self.processed_task_types == {"task_1", "task_2"}:
                self.state = SessionState.COMPLETE
            return "task_1_completed"
        if self.state != SessionState.AUTHORIZED:
            return "task_2_waiting_for_authorization"
        self.current_request_id = str(result["request_id"])
        self.state = SessionState.EXECUTING
        return "task_2_execute"

    def cycle_completed(self) -> None:
        if self.state != SessionState.EXECUTING or self.completed_cycles + self.skipped_cycles >= 6:
            raise SessionError("当前状态不能记录组完成。")
        self.completed_cycles += 1

    def cycle_skipped(self) -> None:
        if self.state != SessionState.EXECUTING or self.completed_cycles + self.skipped_cycles >= 6:
            raise SessionError("当前状态不能记录跳过组。")
        self.skipped_cycles += 1

    def task_2_return_completed(self) -> None:
        if self.state != SessionState.EXECUTING or self.completed_cycles + self.skipped_cycles != 6:
            raise SessionError("六组未全部处理或未在执行中，不能确认任务二完成。")
        self.processed_task_types.add("task_2")
        self.current_request_id = None
        self.state = SessionState.COMPLETE if "task_1" in self.processed_task_types else SessionState.AUTHORIZED

    def revoke(self, reason: str, *, suction_state: str = "unknown") -> None:
        self.state = SessionState.STOPPED
        self.authorization_revoked_reason = reason
        self.suction_state = suction_state
        self.current_request_id = None
