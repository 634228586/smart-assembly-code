from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .vision_client import VisionTargetNotFoundError


class AssemblyError(RuntimeError):
    pass


class RobotPort(Protocol):
    def current_tcp_pose(self) -> tuple[float, ...]: ...
    def pose_trans(self, base_pose, tool_delta) -> tuple[float, ...]: ...
    def move_joint(self, target) -> None: ...
    def move_line(self, target) -> None: ...
    def set_suction(self, enabled: bool) -> None: ...
    def request_stop(self) -> None: ...


class VisionPort(Protocol):
    def locate_block(self, *, request_id: str, color: str, photo_point: str, session_id: str) -> dict[str, Any]: ...
    def locate_trays(self, *, request_id: str, photo_point: str, session_id: str) -> dict[str, dict[str, Any]]: ...


@dataclass(frozen=True)
class AssemblyResult:
    processed_cycles: int
    completed_cycles: int
    skipped_cycles: int
    skipped: tuple[dict[str, Any], ...]
    returned_to_task_card: bool
    suction_may_be_on: bool


def normalize_square_angle(value: float) -> float:
    half_pi = math.pi / 2
    return (float(value) + math.pi / 4) % half_pi - math.pi / 4


class AssemblyExecutor:
    """同步执行核心；正式 Qt层必须把它放到独占 Robot Worker线程。"""

    def __init__(self, robot: RobotPort, vision: VisionPort, *, session_id: str, points: dict[str, list[float]], reference_anchors: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None, cycle_completed: Callable[[], None] | None = None, stop_event: threading.Event | None = None) -> None:
        self.robot = robot; self.vision = vision; self.session_id = session_id
        self.points = points; self.reference_anchors = reference_anchors
        self.progress = progress or (lambda event: None)
        self.on_cycle_completed = cycle_completed or (lambda: None)
        self.stop_event = stop_event or threading.Event(); self.suction_may_be_on = False

    def request_stop(self) -> None:
        self.stop_event.set()
        self.robot.request_stop()

    def _guard(self) -> None:
        if self.stop_event.is_set():
            raise AssemblyError("收到人工停止请求。")

    def _emit(self, cycle: int, phase: str, message: str) -> None:
        self.progress({"cycle": cycle, "phase": phase, "message": message})

    def _capture_target_with_one_retry(self, cycle: int, phase: str, operation):
        try:
            return operation(1)
        except VisionTargetNotFoundError as first_error:
            self._guard()
            self._emit(cycle, phase, f"首次未找到目标，机器人保持停稳并重拍一次：{first_error}")
            try:
                return operation(2)
            except VisionTargetNotFoundError:
                return None

    def _capture_trays_with_one_retry(self, required_colors: set[str]) -> dict[str, dict[str, Any]]:
        trays = self.vision.locate_trays(
            request_id=f"{self.session_id}-T-A1",
            photo_point="trays_photo",
            session_id=self.session_id,
        )
        missing = required_colors - set(trays)
        if not missing:
            return trays
        self._guard()
        self._emit(0, "tray_capture_retry", f"托盘缺少{'、'.join(sorted(missing))}，机器人保持停稳并重拍一次")
        retry = self.vision.locate_trays(
            request_id=f"{self.session_id}-T-A2",
            photo_point="trays_photo",
            session_id=self.session_id,
        )
        trays.update(retry)
        return trays

    @staticmethod
    def _with_z(pose: tuple[float, ...], z: float) -> tuple[float, ...]:
        value = list(pose); value[2] = float(z); return tuple(value)

    def _reference_anchor(self, scene: str, color: str) -> tuple[float, ...]:
        scene_anchors = self.reference_anchors.get(scene)
        anchor = scene_anchors.get(color) if isinstance(scene_anchors, dict) else None
        if not isinstance(anchor, (list, tuple)) or len(anchor) != 6:
            action = "抓取" if scene == "blocks" else "放置"
            raise AssemblyError(f"{scene}/{color}色基准{action}点尚未示教；禁止运动和吸盘动作。")
        try:
            values = tuple(float(value) for value in anchor)
        except (TypeError, ValueError) as exc:
            raise AssemblyError(f"{scene}/{color}色基准点包含非数值。") from exc
        if not all(math.isfinite(value) for value in values):
            raise AssemblyError(f"{scene}/{color}色基准点包含无效数值。")
        return values

    def _validate_required_anchors(self, pairs: list[tuple[str, str]]) -> None:
        for scene, color in pairs:
            self._reference_anchor(scene, color)

    @staticmethod
    def _reference_high_pose(anchor: tuple[float, ...], photo_pose: tuple[float, ...]) -> tuple[float, ...]:
        # The operator may teach close to the object.  Preserve the taught X/Y
        # and orientation, but always approach horizontally at the photo-safe Z.
        return (float(anchor[0]), float(anchor[1]), float(photo_pose[2]), float(anchor[3]), float(anchor[4]), float(anchor[5]))

    @staticmethod
    def _apply_calibrated_xy(reference_pose: tuple[float, ...], delta_x_m: Any, delta_y_m: Any) -> tuple[float, ...]:
        """Apply the eye-in-hand nine-point result in robot-base X/Y.

        During calibration the camera moves while the target stays fixed.  The
        stored pixel mapping therefore reports the camera displacement that
        would reproduce the observed pixel displacement.  A target that moves
        in the live image requires the equal and opposite robot-base movement.
        These values must not be passed to poseTrans as tool-frame X/Y; doing so
        rotates/swaps the correction a second time.
        """
        dx, dy = float(delta_x_m), float(delta_y_m)
        return (
            float(reference_pose[0]) - dx,
            float(reference_pose[1]) - dy,
            *tuple(float(value) for value in reference_pose[2:]),
        )

    def _run_cycle(
        self,
        *,
        cycle: int,
        block_color: str,
        tray_color: str,
        tray: dict[str, Any],
        tray_photo: tuple[float, ...],
    ) -> bool:
        self._guard()
        self._emit(cycle, "block_photo_point", "直达方块拍照点")
        self.robot.move_joint(self.points["blocks_photo"])
        block_photo = self.robot.current_tcp_pose()
        self._emit(cycle, "block_capture", "采集当前颜色方块")
        block = self._capture_target_with_one_retry(cycle, "block_capture_retry", lambda attempt: self.vision.locate_block(request_id=f"{self.session_id}-B{cycle}-A{attempt}", color=block_color, photo_point="blocks_photo", session_id=self.session_id))
        if block is None:
            self._emit(cycle, "cycle_skipped", f"连续两帧未找到{block_color}色方块，本组跳过且未开启吸盘")
            return False
        self._emit(
            cycle,
            "block_visual_offset",
            f"{block_color}色方块：相对本色物理基准的局部变化"
            f"ΔX={float(block['delta_x_tool_m']) * 1000.0:+.3f} mm，"
            f"ΔY={float(block['delta_y_tool_m']) * 1000.0:+.3f} mm，"
            f"ΔR={math.degrees(float(block['delta_r_rad'])):+.2f}°",
        )
        block_anchor = self._reference_anchor("blocks", block_color)
        block_reference = self._reference_high_pose(block_anchor, block_photo)
        block_target = self._apply_calibrated_xy(block_reference, block["delta_x_tool_m"], block["delta_y_tool_m"])
        self._guard(); self._emit(
            cycle,
            "block_xy",
            f"移动到方块高位目标：基准XY=({block_reference[0]:.6f}, {block_reference[1]:.6f}) m，"
            f"最终XY=({block_target[0]:.6f}, {block_target[1]:.6f}) m",
        )
        self.robot.move_line(block_target)
        self._guard(); self._emit(cycle, "pick_down", f"垂直下降到{block_color}色抓取基准点保存的Z")
        self.robot.move_line(self._with_z(block_target, block_anchor[2]))
        self._guard(); self._emit(cycle, "suction_on", "吸盘吸取并回读")
        self.suction_may_be_on = True; self.robot.set_suction(True)
        self._guard(); self._emit(cycle, "pick_up", "垂直返回方块拍照高度")
        self.robot.move_line(block_target)
        self._emit(cycle, "block_return", "回到方块拍照关节点")
        self.robot.move_joint(self.points["blocks_photo"])

        self._guard(); self._emit(cycle, "tray_photo_point", "直达托盘拍照点")
        self.robot.move_joint(self.points["trays_photo"])
        self._emit(cycle, "tray_cache", "使用抓取前采集的托盘全景缓存")
        self._emit(
            cycle,
            "tray_visual_offset",
            f"{tray_color}色托盘：相对本色物理基准的局部变化"
            f"ΔX={float(tray['delta_x_tool_m']) * 1000.0:+.3f} mm，"
            f"ΔY={float(tray['delta_y_tool_m']) * 1000.0:+.3f} mm，"
            f"ΔR={math.degrees(float(tray['delta_r_rad'])):+.2f}°",
        )
        # The image-plane angle increases opposite to the tool-Z rotation used
        # by poseTrans on this eye-in-hand installation.  Reverse the relative
        # tray/block image angle before commanding the wrist.
        delta_rz = normalize_square_angle(float(block["delta_r_rad"]) - float(tray["delta_r_rad"]))
        tray_anchor = self._reference_anchor("trays", tray_color)
        tray_reference = self._reference_high_pose(tray_anchor, tray_photo)
        tray_xy_target = self._apply_calibrated_xy(tray_reference, tray["delta_x_tool_m"], tray["delta_y_tool_m"])
        tray_target = self.robot.pose_trans(tray_xy_target, (0, 0, 0, 0, 0, delta_rz))
        self._guard(); self._emit(
            cycle,
            "tray_xy_rotation",
            f"移动到托盘高位目标并完成方向对齐：基准XY=({tray_reference[0]:.6f}, {tray_reference[1]:.6f}) m，"
            f"最终XY=({tray_target[0]:.6f}, {tray_target[1]:.6f}) m，"
            f"末端补偿角={math.degrees(delta_rz):+.2f}°",
        )
        self.robot.move_line(tray_target)
        self._guard(); self._emit(cycle, "place_down", f"垂直下降到{tray_color}色放置基准点保存的Z")
        self.robot.move_line(self._with_z(tray_target, tray_anchor[2]))
        self._guard(); self._emit(cycle, "suction_off", "吸盘释放并回读")
        self.robot.set_suction(False); self.suction_may_be_on = False
        self._guard(); self._emit(cycle, "place_up", "垂直返回托盘拍照高度")
        self.robot.move_line(tray_target); self.on_cycle_completed()
        return True

    def run(self, sequence: list[dict[str, Any]]) -> AssemblyResult:
        ordered = sorted(sequence, key=lambda item: item["order"])
        if [item["order"] for item in ordered] != list(range(1, 7)):
            raise AssemblyError("任务二计划未完整覆盖 1..6。")
        self._validate_required_anchors(
            [("blocks", str(item["block_color"])) for item in ordered]
            + [("trays", str(item["tray_color"])) for item in ordered]
        )
        self._guard(); self._emit(0, "tray_photo_point", "抓取前直达托盘拍照点")
        self.robot.move_joint(self.points["trays_photo"])
        tray_photo = self.robot.current_tcp_pose()
        required_trays = {str(item["tray_color"]) for item in ordered}
        self._emit(0, "tray_capture", "抓取前采集托盘全景；缺色时原位重拍一次")
        tray_cache = self._capture_trays_with_one_retry(required_trays)
        completed = 0
        skipped: list[dict[str, Any]] = []
        for item in ordered:
            cycle = int(item["order"])
            block_color = str(item["block_color"]); tray_color = str(item["tray_color"])
            tray = tray_cache.get(tray_color)
            if tray is None:
                self._emit(cycle, "cycle_skipped", f"连续两帧未找到{tray_color}色托盘，本组在抓取前跳过")
                skipped.append({"cycle": cycle, "block_color": block_color, "tray_color": tray_color, "reason": "tray_not_found"})
                continue
            if self._run_cycle(
                cycle=cycle, block_color=block_color, tray_color=tray_color,
                tray=tray, tray_photo=tray_photo,
            ):
                completed += 1
            else:
                skipped.append({"cycle": cycle, "block_color": block_color, "tray_color": tray_color, "reason": "block_not_found"})

        self._guard(); self._emit(6, "return_task_card", "六组完成后返回任务卡拍照点")
        self.robot.move_joint(self.points["task_card_photo"])
        return AssemblyResult(6, completed, len(skipped), tuple(skipped), True, self.suction_may_be_on)

    def run_single(self, *, block_color: str, tray_color: str) -> AssemblyResult:
        allowed = {"红", "橙", "黄", "绿", "蓝", "紫"}
        if block_color not in allowed or tray_color not in allowed:
            raise AssemblyError("单组抓放颜色无效。")
        self._validate_required_anchors([("blocks", block_color), ("trays", tray_color)])
        self._guard(); self._emit(0, "tray_photo_point", "抓取前直达托盘拍照点")
        self.robot.move_joint(self.points["trays_photo"])
        tray_photo = self.robot.current_tcp_pose()
        tray_cache = self._capture_trays_with_one_retry({tray_color})
        skipped: list[dict[str, Any]] = []
        tray = tray_cache.get(tray_color)
        if tray is None:
            self._emit(1, "cycle_skipped", f"连续两帧未找到{tray_color}色托盘，本组在抓取前跳过")
            skipped.append({"cycle": 1, "block_color": block_color, "tray_color": tray_color, "reason": "tray_not_found"})
            completed = 0
        elif self._run_cycle(cycle=1, block_color=block_color, tray_color=tray_color, tray=tray, tray_photo=tray_photo):
            completed = 1
        else:
            completed = 0
            skipped.append({"cycle": 1, "block_color": block_color, "tray_color": tray_color, "reason": "block_not_found"})
        self._guard(); self._emit(1, "direct_return_tray_photo", "单组完成，返回托盘拍照关节点")
        self.robot.move_joint(self.points["trays_photo"])
        self._guard(); self._emit(1, "direct_return_standby", "返回比赛待机点")
        self.robot.move_joint(self.points["competition_standby"])
        return AssemblyResult(1, completed, len(skipped), tuple(skipped), False, self.suction_may_be_on)
