from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Phase(str, Enum):
    BLOCK_PHOTO_POINT = "block_photo_point"
    BLOCK_CAPTURE = "block_capture"
    BLOCK_XY = "block_xy"
    PICK_DOWN = "pick_down"
    SUCTION_ON = "suction_on"
    PICK_UP = "pick_up"
    BLOCK_RETURN = "block_return"
    TRAY_PHOTO_POINT = "tray_photo_point"
    TRAY_CAPTURE_OR_CACHE = "tray_capture_or_cache"
    TRAY_XY_AND_ROTATION = "tray_xy_and_rotation"
    PLACE_DOWN = "place_down"
    SUCTION_OFF = "suction_off"
    PLACE_UP = "place_up"


@dataclass(frozen=True)
class AssemblyStep:
    order: int
    block_color: str
    tray_color: str


@dataclass(frozen=True)
class PlannedPhase:
    cycle: int
    phase: Phase
    block_color: str
    tray_color: str


def build_phases(sequence: Iterable[dict]) -> tuple[PlannedPhase, ...]:
    steps = sorted((AssemblyStep(int(item["order"]), str(item["block_color"]), str(item["tray_color"])) for item in sequence), key=lambda item: item.order)
    if [item.order for item in steps] != list(range(1, 7)):
        raise ValueError("装夹计划必须完整覆盖 1..6。")
    return tuple(PlannedPhase(step.order, phase, step.block_color, step.tray_color) for step in steps for phase in Phase)
