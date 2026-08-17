from __future__ import annotations


def command_matches(instruction: str, phrase: str = "请开始识别任务卡") -> bool:
    """按用户确认的宽松规则：命令短语中的任意单字命中即触发。"""

    if not isinstance(instruction, str) or not instruction.strip():
        return False
    return any(character in instruction for character in phrase if not character.isspace())
