from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


FINAL_TARGET_SETTLE_S = 1.5
FINAL_TARGET_STABLE_READS = 3


class RobotGatewayError(RuntimeError):
    pass


@dataclass(frozen=True)
class RobotSnapshot:
    robot_name: str
    robot_mode: str
    safety_mode: str
    joint_positions: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    tcp_offset: tuple[float, ...]
    exec_id: int
    runtime_state: str


@dataclass(frozen=True)
class MotionPermit:
    session_id: str
    issued_monotonic: float
    config_fingerprint: str
    maximum_age_s: float = 10.0

    def assert_valid(self, expected_fingerprint: str) -> None:
        if not self.session_id or self.config_fingerprint != expected_fingerprint:
            raise RobotGatewayError("运动许可与当前会话或配置不匹配。")
        if time.monotonic() - self.issued_monotonic > self.maximum_age_s:
            raise RobotGatewayError("运动许可已过期。")


def _six_finite(values: Sequence[Any], name: str) -> tuple[float, ...]:
    if len(values) != 6:
        raise RobotGatewayError(f"{name} 必须包含六个数值。")
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise RobotGatewayError(f"{name} 包含非有限数值。")
    return normalized


class AuboRealGateway:
    """真实 AUBO SDK边界。构造不会连接；所有控制方法都要求短时运动许可。"""

    def __init__(self, endpoint: dict[str, Any], robot_config: dict[str, Any]) -> None:
        self.endpoint = endpoint
        self.config = robot_config
        self.client = None
        self.robot_name: str | None = None

    def connect_readonly(self) -> RobotSnapshot:
        if self.client is not None:
            raise RobotGatewayError("机器人客户端已经存在。")
        host, port = self.endpoint.get("host"), self.endpoint.get("port")
        expected_name = self.config["identity"].get("controller_robot_name")
        username = self.config["login"].get("username")
        password = os.environ.get(str(self.config["login"].get("password_env", "")))
        if "UNSET" in {host, port, expected_name, username} or not password:
            raise RobotGatewayError("机器人连接、身份或凭据尚未配置。")
        try:
            import pyaubo_sdk
            client = pyaubo_sdk.RpcClient()
            client.setRequestTimeout(3000)
            if client.connect(host, int(port)) != 0 or not client.hasConnected():
                raise RobotGatewayError("AUBO RPC连接失败。")
            if client.login(username, password) != 0 or not client.hasLogined():
                raise RobotGatewayError("AUBO RPC登录失败。")
            names = [str(value) for value in client.getRobotNames()]
            if names != [expected_name]:
                raise RobotGatewayError(f"控制器机器人身份不匹配：actual={names}, expected={[expected_name]}")
            self.client, self.robot_name = client, expected_name
            return self.snapshot()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        client, self.client, self.robot_name = self.client, None, None
        if client is None:
            return
        try:
            if client.hasLogined(): client.logout()
        finally:
            if client.hasConnected(): client.disconnect()

    def _interfaces(self):
        if self.client is None or self.robot_name is None or not self.client.hasConnected() or not self.client.hasLogined():
            raise RobotGatewayError("机器人未连接并登录。")
        robot = self.client.getRobotInterface(self.robot_name)
        return robot, robot.getRobotState(), robot.getMotionControl()

    def snapshot(self) -> RobotSnapshot:
        # AUBO Python SDK occasionally raises UnicodeDecodeError while decoding
        # a read-only RPC status response.  A single malformed status packet must
        # not turn a successfully completed motion into an assembly failure.
        # Retry only this specific transient decode failure; all robot/safety
        # errors still propagate immediately.
        for attempt in range(3):
            try:
                robot, state, motion = self._interfaces()
                return RobotSnapshot(
                    self.robot_name or "",
                    str(state.getRobotModeType()),
                    str(state.getSafetyModeType()),
                    _six_finite(state.getJointPositions(), "joint_positions"),
                    _six_finite(state.getTcpPose(), "tcp_pose"),
                    _six_finite(robot.getRobotConfig().getTcpOffset(), "tcp_offset"),
                    int(motion.getExecId()),
                    str(self.client.getRuntimeMachine().getStatus()),
                )
            except UnicodeDecodeError as exc:
                if attempt == 2:
                    raise RobotGatewayError("AUBO SDK连续3次无法解码机器人状态响应。") from exc
                time.sleep(0.02)
        raise AssertionError("unreachable")

    def current_tcp_pose(self) -> tuple[float, ...]:
        return self.snapshot().tcp_pose

    def assert_maintenance_gate(self, permit: MotionPermit, fingerprint: str) -> RobotSnapshot:
        """通用维护门控；不要求TCP、Runtime或比赛视觉/标定就绪。"""

        permit.assert_valid(fingerprint)
        snapshot = self.snapshot()
        required = self.config["required_state"]
        if not snapshot.robot_mode.endswith(str(required["robot_mode"])):
            raise RobotGatewayError(f"RobotMode不满足：{snapshot.robot_mode}")
        if not snapshot.safety_mode.endswith(str(required["safety_mode"])):
            raise RobotGatewayError(f"SafetyMode不满足：{snapshot.safety_mode}")
        if snapshot.exec_id != int(required["exec_id"]):
            raise RobotGatewayError(f"控制器执行队列非空：exec_id={snapshot.exec_id}")
        return snapshot

    def assert_io_gate(self, permit: MotionPermit, fingerprint: str) -> RobotSnapshot:
        return self.assert_maintenance_gate(permit, fingerprint)

    def assert_maintenance_motion_gate(self, permit: MotionPermit, fingerprint: str) -> RobotSnapshot:
        """维护运动门控；控制器只在 RuntimeMachine Running 时接受直接运动。"""

        snapshot = self.assert_runtime_start_gate(permit, fingerprint)
        if not snapshot.runtime_state.endswith("Running"):
            raise RobotGatewayError(
                f"RuntimeMachine未启动：{snapshot.runtime_state}；请先启动 RuntimeMachine。"
            )
        return snapshot

    def assert_runtime_start_gate(self, permit: MotionPermit, fingerprint: str) -> RobotSnapshot:
        """自动 start 前的只读门控；允许 RuntimeMachine 当前为 Stopped。"""

        snapshot = self.assert_maintenance_gate(permit, fingerprint)
        self._assert_active_tcp(snapshot)
        return snapshot

    def _assert_active_tcp(self, snapshot: RobotSnapshot) -> None:
        expected_offset = _six_finite(self.config["active_tcp"]["offset"], "expected_tcp_offset")
        tolerance = float(self.config["active_tcp"]["tolerance"])
        if max(abs(a - b) for a, b in zip(snapshot.tcp_offset, expected_offset)) > tolerance:
            raise RobotGatewayError("控制器活动 TCP offset 与批准配置不一致。")

    def start_runtime_for_maintenance(self, permit: MotionPermit, fingerprint: str, timeout_s: float = 5.0) -> bool:
        """经维护门控后启动 RuntimeMachine；绝不调用 runProgram。返回本次是否执行了启动。"""

        snapshot = self.assert_runtime_start_gate(permit, fingerprint)
        if snapshot.runtime_state.endswith("Running"):
            return False
        if self.client is None:
            raise RobotGatewayError("机器人未连接。")
        runtime = self.client.getRuntimeMachine()
        result = runtime.start()
        if not isinstance(result, int) or result != 0:
            raise RobotGatewayError(f"RuntimeMachine.start失败：{result}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = self.snapshot()
            self._assert_snapshot_safe(current)
            if current.exec_id != int(self.config["required_state"]["exec_id"]):
                raise RobotGatewayError(f"RuntimeMachine启动后执行队列非空：exec_id={current.exec_id}")
            if current.runtime_state.endswith("Running"):
                return True
            time.sleep(0.05)
        raise RobotGatewayError("RuntimeMachine.start返回成功，但状态未在超时内变为 Running。")

    def assert_motion_gate(self, permit: MotionPermit, fingerprint: str) -> RobotSnapshot:
        snapshot = self.assert_maintenance_motion_gate(permit, fingerprint)
        required = self.config["required_state"]
        if required.get("runtime_machine_running") is True and not snapshot.runtime_state.endswith("Running"):
            raise RobotGatewayError(f"RuntimeMachine不满足：{snapshot.runtime_state}")
        return snapshot

    def _assert_snapshot_safe(self, snapshot: RobotSnapshot) -> None:
        required = self.config["required_state"]
        if not snapshot.robot_mode.endswith(str(required["robot_mode"])) or not snapshot.safety_mode.endswith(str(required["safety_mode"])):
            try:
                self.stop_motion()
            finally:
                raise RobotGatewayError(f"运动期间安全状态改变：RobotMode={snapshot.robot_mode}, SafetyMode={snapshot.safety_mode}")

    def pose_trans(self, base_pose: Sequence[Any], tool_delta: Sequence[Any]) -> tuple[float, ...]:
        if self.client is None:
            raise RobotGatewayError("机器人未连接。")
        return _six_finite(self.client.getMath().poseTrans(list(_six_finite(base_pose, "base_pose")), list(_six_finite(tool_delta, "tool_delta"))), "target_pose")

    def pose_trans_inv(self, base_pose: Sequence[Any], target_pose: Sequence[Any]) -> tuple[float, ...]:
        """Return ``target_pose`` expressed relative to ``base_pose`` using the controller math API."""

        if self.client is None:
            raise RobotGatewayError("机器人未连接。")
        return _six_finite(
            self.client.getMath().poseTransInv(
                list(_six_finite(base_pose, "base_pose")),
                list(_six_finite(target_pose, "target_pose")),
            ),
            "relative_pose",
        )

    def _issue_move_joint(self, target: Sequence[Any], limits: dict[str, Any]) -> int:
        _, _, motion = self._interfaces()
        speed_fraction = float(limits["speed_fraction"])
        maximum = float(limits["maximum_authorized_speed_fraction"])
        if not 0 < speed_fraction <= maximum <= 1:
            raise RobotGatewayError("速度比例超过已批准上限或配置无效。")
        if motion.setSpeedFraction(speed_fraction) != 0:
            raise RobotGatewayError("setSpeedFraction失败。")
        result = motion.moveJoint(list(_six_finite(target, "joint_target")), float(limits["joint_acceleration_rad_s2"]), float(limits["joint_velocity_rad_s"]), 0.0, 0.0)
        if not isinstance(result, int) or result != 0:
            if result == 13:
                raise RobotGatewayError("moveJoint被控制器忽略：AUBO_REQUEST_IGNORE=13；程序已尝试自动启动 RuntimeMachine，请检查控制器状态。")
            raise RobotGatewayError(f"moveJoint失败：{result}")
        return result

    def move_joint(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str) -> int:
        self.start_runtime_for_maintenance(permit, fingerprint)
        self.assert_motion_gate(permit, fingerprint)
        return self._issue_move_joint(target, limits)

    def move_joint_maintenance(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str) -> int:
        self.start_runtime_for_maintenance(permit, fingerprint)
        self.assert_maintenance_motion_gate(permit, fingerprint)
        return self._issue_move_joint(target, limits)

    def move_line(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str) -> int:
        self.start_runtime_for_maintenance(permit, fingerprint)
        self.assert_motion_gate(permit, fingerprint)
        return self._issue_move_line(target, limits)

    def move_line_maintenance(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str) -> int:
        self.start_runtime_for_maintenance(permit, fingerprint)
        self.assert_maintenance_motion_gate(permit, fingerprint)
        return self._issue_move_line(target, limits)

    def _issue_move_line(self, target: Sequence[Any], limits: dict[str, Any]) -> int:
        _, _, motion = self._interfaces()
        speed_fraction = float(limits["speed_fraction"])
        maximum = float(limits["maximum_authorized_speed_fraction"])
        if not 0 < speed_fraction <= maximum <= 1:
            raise RobotGatewayError("速度比例超过已批准上限或配置无效。")
        if motion.setSpeedFraction(speed_fraction) != 0:
            raise RobotGatewayError("setSpeedFraction失败。")
        result = motion.moveLine(list(_six_finite(target, "tcp_target")), float(limits["linear_acceleration_m_s2"]), float(limits["linear_velocity_m_s"]), 0.0, 0.0)
        if not isinstance(result, int) or result != 0:
            raise RobotGatewayError(f"moveLine失败：{result}")
        return result

    def _target_reached(self, actual: tuple[float, ...], target: tuple[float, ...], *, kind: str, position_tolerance: float, orientation_tolerance: float) -> bool:
        if kind == "joint":
            return max(abs(a - b) for a, b in zip(actual, target)) <= position_tolerance
        if self.client is None:
            raise RobotGatewayError("机器人未连接，无法计算TCP姿态误差。")
        position_ok = max(abs(actual[index] - target[index]) for index in range(3)) <= position_tolerance
        orientation_error = float(self.client.getMath().poseAngleDistance(list(actual), list(target)))
        if not math.isfinite(orientation_error):
            raise RobotGatewayError("AUBO姿态距离计算返回非有限数值。")
        return position_ok and orientation_error <= orientation_tolerance

    def _target_error_details(self, actual: tuple[float, ...], target: tuple[float, ...], *, kind: str) -> str:
        delta = tuple(actual_value - target_value for actual_value, target_value in zip(actual, target))
        if kind == "joint":
            return (
                f"目标关节={list(target)}，实际关节={list(actual)}，"
                f"Δ关节(rad)={list(delta)}，最大绝对误差={max(abs(value) for value in delta):.6f} rad"
            )
        if self.client is None:
            raise RobotGatewayError("机器人未连接，无法计算TCP姿态误差。")
        position_delta_mm = tuple(value * 1000.0 for value in delta[:3])
        orientation_error = float(self.client.getMath().poseAngleDistance(list(actual), list(target)))
        if not math.isfinite(orientation_error):
            raise RobotGatewayError("AUBO姿态距离计算返回非有限数值。")
        return (
            f"目标TCP={list(target)}，实际TCP={list(actual)}，"
            f"Δ位置(mm)={list(position_delta_mm)}，"
            f"位置最大绝对误差={max(abs(value) for value in position_delta_mm):.3f} mm，"
            f"AUBO真实姿态误差={math.degrees(orientation_error):.3f}°"
        )

    def _wait_until_target(self, target: tuple[float, ...], *, kind: str, position_tolerance: float, orientation_tolerance: float, timeout_s: float, should_stop: Callable[[], bool] | None = None) -> None:
        _, _, motion = self._interfaces()
        stop_requested = should_stop or (lambda: False)
        start_deadline = time.monotonic() + min(5.0, timeout_s)
        finish_deadline = time.monotonic() + timeout_s
        observed_running = False
        # AUBO的move API返回成功后，exec_id可能延迟几十毫秒才出现。
        # 即使当前位置已经落在目标容差内，也必须持续观察一段时间，
        # 防止把“命令尚未入队”误判为“命令已经执行完成”。
        idle_target_since: float | None = None
        enqueue_observation_s = 1.0
        final_target_settle_deadline: float | None = None
        final_target_stable_reads = 0
        while time.monotonic() < start_deadline:
            if stop_requested():
                self.stop_motion()
                raise RobotGatewayError("收到人工停止请求，已向控制器发送停止命令。")
            if int(motion.getExecId()) != -1:
                observed_running = True
                break
            snapshot = self.snapshot()
            self._assert_snapshot_safe(snapshot)
            actual = snapshot.joint_positions if kind == "joint" else snapshot.tcp_pose
            if self._target_reached(actual, target, kind=kind, position_tolerance=position_tolerance, orientation_tolerance=orientation_tolerance):
                if idle_target_since is None:
                    idle_target_since = time.monotonic()
                elif time.monotonic() - idle_target_since >= enqueue_observation_s:
                    if int(motion.getExecId()) == -1:
                        return
            else:
                idle_target_since = None
            time.sleep(0.05)
        if not observed_running:
            raise RobotGatewayError("运动未在开始超时内启动且目标误差不满足。")
        while time.monotonic() < finish_deadline:
            if stop_requested():
                self.stop_motion()
                raise RobotGatewayError("收到人工停止请求，已向控制器发送停止命令。")
            snapshot = self.snapshot()
            self._assert_snapshot_safe(snapshot)
            if snapshot.exec_id == -1:
                actual = snapshot.joint_positions if kind == "joint" else snapshot.tcp_pose
                now = time.monotonic()
                if final_target_settle_deadline is None:
                    final_target_settle_deadline = min(finish_deadline, now + FINAL_TARGET_SETTLE_S)
                if self._target_reached(actual, target, kind=kind, position_tolerance=position_tolerance, orientation_tolerance=orientation_tolerance):
                    final_target_stable_reads += 1
                    if final_target_stable_reads >= FINAL_TARGET_STABLE_READS:
                        return
                else:
                    final_target_stable_reads = 0
                # The controller can mark its queue idle just before the final
                # read-only pose packet reflects the settled servo position.
                # Require several consecutive in-tolerance packets and allow a
                # short convergence window before treating a residual as real.
                if now >= final_target_settle_deadline:
                    details = self._target_error_details(actual, target, kind=kind)
                    raise RobotGatewayError("运动结束后最终位置误差超限（已等待状态稳定）：" + details)
            else:
                final_target_settle_deadline = None
                final_target_stable_reads = 0
            time.sleep(0.05)
        raise RobotGatewayError("运动完成等待超时；不会自动继续下一阶段。")

    def move_joint_and_wait(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str, timeout_s: float = 180.0, should_stop: Callable[[], bool] | None = None) -> None:
        normalized = _six_finite(target, "joint_target")
        self.move_joint(normalized, limits, permit, fingerprint)
        tolerance = float(limits["joint_tolerance_rad"])
        self._wait_until_target(normalized, kind="joint", position_tolerance=tolerance, orientation_tolerance=tolerance, timeout_s=timeout_s, should_stop=should_stop)

    def move_joint_maintenance_and_wait(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str, timeout_s: float = 180.0, should_stop: Callable[[], bool] | None = None) -> None:
        normalized = _six_finite(target, "joint_target")
        self.move_joint_maintenance(normalized, limits, permit, fingerprint)
        tolerance = float(limits["joint_tolerance_rad"])
        self._wait_until_target(normalized, kind="joint", position_tolerance=tolerance, orientation_tolerance=tolerance, timeout_s=timeout_s, should_stop=should_stop)

    def move_line_and_wait(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str, timeout_s: float = 180.0, should_stop: Callable[[], bool] | None = None) -> None:
        normalized = _six_finite(target, "tcp_target")
        self.move_line(normalized, limits, permit, fingerprint)
        self._wait_until_target(
            normalized,
            kind="tcp",
            position_tolerance=float(limits["tcp_position_tolerance_m"]),
            orientation_tolerance=float(limits["tcp_orientation_tolerance_rad"]),
            timeout_s=timeout_s,
            should_stop=should_stop,
        )

    def move_line_maintenance_and_wait(self, target: Sequence[Any], limits: dict[str, Any], permit: MotionPermit, fingerprint: str, timeout_s: float = 180.0, should_stop: Callable[[], bool] | None = None) -> None:
        normalized = _six_finite(target, "tcp_target")
        self.move_line_maintenance(normalized, limits, permit, fingerprint)
        self._wait_until_target(
            normalized,
            kind="tcp",
            position_tolerance=float(limits["tcp_position_tolerance_m"]),
            orientation_tolerance=float(limits["tcp_orientation_tolerance_rad"]),
            timeout_s=timeout_s,
            should_stop=should_stop,
        )

    def set_suction(self, enabled: bool, io_config: dict[str, Any], permit: MotionPermit, fingerprint: str, should_stop: Callable[[], bool] | None = None) -> bool:
        self.start_runtime_for_maintenance(permit, fingerprint)
        self.assert_motion_gate(permit, fingerprint)
        return self._set_tool_suction(enabled, io_config, should_stop=should_stop)

    def set_suction_maintenance(self, enabled: bool, io_config: dict[str, Any], permit: MotionPermit, fingerprint: str, should_stop: Callable[[], bool] | None = None) -> bool:
        """维护页使用基础IO门控操作工具吸盘，不要求Runtime/TCP/视觉就绪。"""

        self.assert_io_gate(permit, fingerprint)
        return self._set_tool_suction(enabled, io_config, should_stop=should_stop)

    def _set_tool_suction(self, enabled: bool, io_config: dict[str, Any], should_stop: Callable[[], bool] | None = None) -> bool:
        stop_requested = should_stop or (lambda: False)
        if stop_requested():
            raise RobotGatewayError("收到人工停止请求，未写入吸盘 IO。")
        if io_config.get("output_type") != "tool_digital_output":
            raise RobotGatewayError("真实吸盘必须配置为工具数字输出。")
        index, enable_index = io_config.get("output_index"), io_config.get("enable_output_index")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (index, enable_index)) or index == enable_index:
            raise RobotGatewayError("吸盘工具控制口或长期使能口无效。")
        level = io_config.get("suction_level" if enabled else "release_level")
        enable_level = io_config.get("enable_level")
        if not isinstance(level, bool) or not isinstance(enable_level, bool):
            raise RobotGatewayError("真实吸盘控制电平或长期使能电平尚未配置。")
        robot, _, _ = self._interfaces()
        io_control = robot.getIoControl()
        for port, port_level, label in ((enable_index, enable_level, "吸盘长期使能 TOOL_IO"), (index, level, "吸盘控制 TOOL_IO")):
            if bool(io_control.isToolIoInput(port)):
                result = io_control.setToolIoInput(port, False)
                if not isinstance(result, int) or result != 0 or bool(io_control.isToolIoInput(port)):
                    raise RobotGatewayError(f"{label}[{port}]切换为输出模式失败：{result}")
            result = io_control.setToolDigitalOutput(port, port_level)
            if not isinstance(result, int) or result != 0:
                raise RobotGatewayError(f"{label}[{port}]写入失败：{result}")
        deadline = time.monotonic() + float(io_config["confirmation_timeout_s"])
        while time.monotonic() < deadline:
            if stop_requested():
                raise RobotGatewayError("吸盘 IO 写入后收到人工停止；实际吸盘状态需人工核对。")
            output_ok = (
                bool(io_control.getToolDigitalOutput(enable_index)) == enable_level
                and bool(io_control.getToolDigitalOutput(index)) == level
            )
            feedback_mode = io_config.get("feedback_mode")
            if feedback_mode == "none":
                feedback_ok = True
            elif feedback_mode == "standard_digital_input":
                feedback_index = io_config.get("feedback_index")
                feedback_level = io_config.get("feedback_active_level")
                if isinstance(feedback_index, bool) or not isinstance(feedback_index, int) or feedback_index < 0 or not isinstance(feedback_level, bool):
                    raise RobotGatewayError("吸盘真空反馈输入配置无效。")
                expected_feedback = feedback_level if enabled else not feedback_level
                feedback_ok = bool(io_control.getStandardDigitalInput(feedback_index)) == expected_feedback
            else:
                raise RobotGatewayError("吸盘 feedback_mode 必须为 none 或 standard_digital_input。")
            if output_ok and feedback_ok:
                return True
            time.sleep(0.05)
        raise RobotGatewayError("吸盘输出或真空反馈回读超时；实际状态未知。")

    def set_standard_digital_output(self, *, index: int, level: bool, permit: MotionPermit, fingerprint: str, should_stop: Callable[[], bool] | None = None, label: str = "数字输出", confirmation_timeout_s: float = 2.0) -> bool:
        """在独立维护IO门控下写入标准DO，并以控制器回读作为成功依据。"""

        self.assert_io_gate(permit, fingerprint)
        stop_requested = should_stop or (lambda: False)
        if stop_requested():
            raise RobotGatewayError(f"收到人工停止请求，未写入{label}。")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RobotGatewayError(f"{label}编号无效。")
        if not isinstance(level, bool):
            raise RobotGatewayError(f"{label}电平无效。")
        if not math.isfinite(float(confirmation_timeout_s)) or float(confirmation_timeout_s) <= 0:
            raise RobotGatewayError(f"{label}回读确认超时配置无效。")
        robot, _, _ = self._interfaces()
        io_control = robot.getIoControl()
        result = io_control.setStandardDigitalOutput(index, level)
        if not isinstance(result, int) or result != 0:
            raise RobotGatewayError(f"{label}写入失败：{result}")
        if stop_requested():
            raise RobotGatewayError(f"{label}写入后收到人工停止；实际状态需人工核对。")
        deadline = time.monotonic() + float(confirmation_timeout_s)
        actual = bool(io_control.getStandardDigitalOutput(index))
        while actual != level and time.monotonic() < deadline:
            if stop_requested():
                raise RobotGatewayError(f"{label}写入后收到人工停止；实际状态需人工核对。")
            time.sleep(0.05)
            actual = bool(io_control.getStandardDigitalOutput(index))
        if actual != level:
            raise RobotGatewayError(f"{label}回读超时：requested={level}, actual={actual}")
        return actual

    def toggle_standard_digital_output(self, *, index: int, on_level: bool, off_level: bool, permit: MotionPermit, fingerprint: str, should_stop: Callable[[], bool] | None = None, label: str = "数字输出", confirmation_timeout_s: float = 2.0) -> bool:
        """读取当前 DO 后切换状态；返回值表示切换后是否处于“开”状态。"""

        self.assert_io_gate(permit, fingerprint)
        if on_level is off_level or not isinstance(on_level, bool) or not isinstance(off_level, bool):
            raise RobotGatewayError(f"{label}开/关电平配置无效。")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RobotGatewayError(f"{label}编号无效。")
        robot, _, _ = self._interfaces()
        current_level = bool(robot.getIoControl().getStandardDigitalOutput(index))
        target_on = current_level != on_level
        target_level = on_level if target_on else off_level
        self.set_standard_digital_output(
            index=index,
            level=target_level,
            permit=permit,
            fingerprint=fingerprint,
            should_stop=should_stop,
            label=label,
            confirmation_timeout_s=confirmation_timeout_s,
        )
        return target_on

    def stop_motion(self) -> None:
        _, _, motion = self._interfaces()
        joint_result = motion.stopJoint(1.0)
        line_result = motion.stopLine(1.0)
        if joint_result not in (0, None) and line_result not in (0, None):
            raise RobotGatewayError(f"停止请求失败：joint={joint_result}, line={line_result}")
