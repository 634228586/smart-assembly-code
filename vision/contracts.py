from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CameraContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureRequest:
    request_id: str
    session_dir: Path
    profile: str

    def validate(self) -> None:
        if self.profile not in {"task_card", "blocks", "trays"}:
            raise CameraContractError("未知相机配置角色。")
        if not self.request_id:
            raise CameraContractError("拍照请求号缺失。")
        if not self.session_dir.is_absolute():
            raise CameraContractError("session目录必须是绝对路径。")


@dataclass(frozen=True)
class CapturedFrame:
    request_id: str
    profile: str
    image_path: Path
    captured_at: str
    parameters_applied: bool

    def validate_for(self, request: CaptureRequest) -> None:
        request.validate()
        if self.request_id != request.request_id or self.profile != request.profile:
            raise CameraContractError("帧身份与本次拍照请求不匹配。")
        if not self.parameters_applied or not self.image_path.is_file():
            raise CameraContractError("相机参数未成功写入或图片未保存。")
        try:
            self.image_path.resolve().relative_to(request.session_dir.resolve())
        except ValueError as exc:
            raise CameraContractError("图片不属于本场 session。") from exc
