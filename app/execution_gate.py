from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from .session import CompetitionSession, SessionState


class ExecutionGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingExecution:
    request_id: str
    token: str
    config_fingerprint: str
    expires_monotonic: float
    recognition: dict[str, Any]


class TwoPhaseExecutionGate:
    """识别结果先校验保存，再由一次性短时令牌提交真实装夹。"""

    def __init__(self, session: CompetitionSession, *, token_lifetime_s: float = 120.0) -> None:
        self.session = session; self.token_lifetime_s = token_lifetime_s; self.pending: PendingExecution | None = None

    def prepare(self, recognition: dict[str, Any], config_fingerprint: str) -> dict[str, Any]:
        outcome = self.session.accept_recognition(recognition)
        if outcome != "task_2_execute":
            return {"type": "recognition_outcome", "request_id": recognition["request_id"], "status": outcome}
        token = secrets.token_urlsafe(32)
        self.pending = PendingExecution(recognition["request_id"], token, config_fingerprint, time.monotonic() + self.token_lifetime_s, recognition)
        return {"type": "ready_to_execute", "request_id": recognition["request_id"], "execute_token": token, "expires_in_s": self.token_lifetime_s}

    def commit(self, *, request_id: str, execute_token: str, config_fingerprint: str) -> dict[str, Any]:
        pending = self.pending
        self.pending = None
        if pending is None:
            raise ExecutionGateError("不存在待提交的任务二。")
        if time.monotonic() > pending.expires_monotonic:
            self.session.revoke("执行令牌超时")
            raise ExecutionGateError("执行令牌已超时，比赛授权已撤销。")
        if pending.request_id != request_id or not secrets.compare_digest(pending.token, execute_token):
            self.session.revoke("执行令牌不匹配")
            raise ExecutionGateError("执行令牌不匹配，比赛授权已撤销。")
        if pending.config_fingerprint != config_fingerprint:
            self.session.revoke("配置在提交前改变")
            raise ExecutionGateError("配置指纹改变，比赛授权已撤销。")
        if self.session.state != SessionState.EXECUTING:
            raise ExecutionGateError("比赛会话不在执行状态。")
        return pending.recognition

    def cancel(self, reason: str) -> None:
        self.pending = None
        self.session.revoke(reason)
