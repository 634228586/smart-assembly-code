from __future__ import annotations

import ctypes
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MVS_WRAPPER_DIR = PACKAGE_DIR / "vendor" / "mvs"
MVS_RUNTIME_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
    Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"),
)
SINGLE_CAPTURE_MAX_ATTEMPTS = 3
RETRYABLE_FRAME_CODES = frozenset({"EMPTY_FRAME", "INVALID_FRAME"})


class MvsCameraError(RuntimeError):
    def __init__(self, code: str, message: str, sdk_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.sdk_code = sdk_code


@dataclass(frozen=True)
class MvsDevice:
    index: int
    model: str
    transport: str


@dataclass(frozen=True)
class MvsCapture:
    image_bgr: np.ndarray
    frame_number: int
    parameters_applied: bool
    configured_parameters: dict[str, Any] = field(default_factory=dict)


def _unsigned(value: object) -> int:
    return int(value) & 0xFFFFFFFF


def _decode(value: object) -> str:
    try:
        raw = memoryview(value).tobytes().split(b"\0", 1)[0]
    except (TypeError, ValueError):
        return ""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def installed_runtime_dir() -> Path | None:
    return next((path for path in MVS_RUNTIME_CANDIDATES if (path / "MvCameraControl.dll").is_file()), None)


def sdk_files_available() -> bool:
    required = ("MvCameraControl_class.py", "CameraParams_header.py", "PixelType_header.py")
    return installed_runtime_dir() is not None and all((MVS_WRAPPER_DIR / name).is_file() for name in required)


class MvsCamera:
    """单台 MVS相机的窄边界；自动打开枚举到的第一台设备。"""

    def __init__(self, sdk: ModuleType | Any | None = None) -> None:
        self._runtime_dir = installed_runtime_dir()
        self._dll_handle: Any | None = None
        self._path_before: str | None = None
        self._injected_wrapper = False
        self._sdk = sdk or self._load_sdk()
        self._camera: Any | None = None
        self._device_list: Any | None = None
        self._devices: list[MvsDevice] = []
        self._opened = False
        self._grabbing = False
        self._software_trigger_configured = False
        self._initialized = False
        # MV_CC_Initialize/Finalize 是新版 Runtime才提供的全局生命周期接口。
        # 旧版 DLL不导出这对符号，但枚举、创建句柄和采集接口仍可正常使用。
        if self._sdk_exports("MV_CC_Initialize"):
            self._check(self._sdk.MvCamera.MV_CC_Initialize(), "SDK_INIT_FAILED", "MVS SDK初始化失败。")
            self._initialized = True

    def _load_sdk(self) -> ModuleType:
        if not sdk_files_available():
            raise MvsCameraError("SDK_NOT_AVAILABLE", "包内 MVS Python wrapper或机器级 Runtime缺失。")
        assert self._runtime_dir is not None
        self._path_before = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self._runtime_dir) + os.pathsep + self._path_before
        if hasattr(os, "add_dll_directory"):
            self._dll_handle = os.add_dll_directory(str(self._runtime_dir))
        wrapper = str(MVS_WRAPPER_DIR)
        if wrapper not in sys.path:
            sys.path.insert(0, wrapper)
            self._injected_wrapper = True
        try:
            return importlib.import_module("MvCameraControl_class")
        except Exception as exc:
            self._release_loader()
            raise MvsCameraError("SDK_LOAD_FAILED", f"MVS SDK加载失败：{exc}") from exc

    def enumerate_devices(self) -> list[MvsDevice]:
        device_list = self._sdk.MV_CC_DEVICE_INFO_LIST()
        mask = 0
        for name in ("MV_GIGE_DEVICE", "MV_USB_DEVICE", "MV_GENTL_GIGE_DEVICE", "MV_GENTL_CAMERALINK_DEVICE", "MV_GENTL_CXP_DEVICE", "MV_GENTL_XOF_DEVICE"):
            mask |= int(getattr(self._sdk, name, 0))
        self._check(self._sdk.MvCamera.MV_CC_EnumDevices(mask, device_list), "ENUM_FAILED", "MVS相机枚举失败。")
        self._device_list = device_list
        self._devices = [self._describe(index, self._device_info(index)) for index in range(int(device_list.nDeviceNum))]
        return list(self._devices)

    def open_first_available(self) -> MvsDevice:
        devices = self.enumerate_devices()
        if not devices:
            raise MvsCameraError("CAMERA_NOT_FOUND", "未枚举到可用的 MVS相机。")
        selected = devices[0]
        self._camera = self._sdk.MvCamera()
        try:
            self._check(self._camera.MV_CC_CreateHandle(self._device_info(selected.index)), "CREATE_HANDLE_FAILED", "创建 MVS相机句柄失败。")
            self._check(self._camera.MV_CC_OpenDevice(self._sdk.MV_ACCESS_Exclusive, 0), "OPEN_FAILED", "独占打开 MVS相机失败。")
            self._opened = True
            self._software_trigger_configured = False
        except Exception:
            self._cleanup_camera()
            raise
        return selected

    def capture(self, profile: dict[str, Any], *, timeout_ms: int = 3000, require_approved: bool = True) -> MvsCapture:
        if not self._opened or self._camera is None:
            raise MvsCameraError("NOT_OPEN", "MVS相机尚未打开。")
        if self._grabbing:
            self._check(self._camera.MV_CC_StopGrabbing(), "STOP_FAILED", "停止 MVS取流失败。")
            self._grabbing = False
        self._apply_profile(profile, require_approved=require_approved)
        self._check(self._camera.MV_CC_StartGrabbing(), "START_FAILED", "启动 MVS取流失败。")
        self._grabbing = True
        try:
            for attempt in range(1, SINGLE_CAPTURE_MAX_ATTEMPTS + 1):
                self._check(self._camera.MV_CC_SetCommandValue("TriggerSoftware"), "TRIGGER_FAILED", "MVS软件触发失败。")
                frame = self._sdk.MV_FRAME_OUT()
                ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
                ret = self._camera.MV_CC_GetImageBuffer(frame, int(timeout_ms))
                self._check(ret, "CAPTURE_TIMEOUT", "MVS拍照超时或取帧失败。")
                try:
                    if not frame.pBufAddr:
                        raise MvsCameraError("EMPTY_FRAME", "MVS返回空图像缓冲区。")
                    image = self._frame_to_bgr(frame)
                    frame_number = int(frame.stFrameInfo.nFrameNum)
                    break
                except MvsCameraError as exc:
                    if exc.code not in RETRYABLE_FRAME_CODES or attempt == SINGLE_CAPTURE_MAX_ATTEMPTS:
                        raise
                finally:
                    self._check(self._camera.MV_CC_FreeImageBuffer(frame), "FREE_BUFFER_FAILED", "释放 MVS图像缓冲区失败。")
        finally:
            # Always leave the camera idle, including after three bad frames, so
            # the next request can safely apply its profile and trigger again.
            if self._grabbing:
                self._check(self._camera.MV_CC_StopGrabbing(), "STOP_FAILED", "单帧完成后停止 MVS取流失败。")
                self._grabbing = False
        return MvsCapture(image, frame_number, True, self._configured_parameters(profile))

    def start_preview(self, profile: dict[str, Any], *, require_approved: bool = True) -> None:
        """按保存的 profile 启动连续取流，仅供实时画面预览。"""

        if not self._opened or self._camera is None:
            raise MvsCameraError("NOT_OPEN", "MVS相机尚未打开。")
        if self._grabbing:
            self.stop_preview()
        self._apply_profile(profile, require_approved=require_approved)
        self._set_enum("TriggerMode", "Off")
        self._software_trigger_configured = False
        self._check(self._camera.MV_CC_StartGrabbing(), "START_FAILED", "启动 MVS连续预览失败。")
        self._grabbing = True

    def read_preview_frame(self, *, timeout_ms: int = 1000) -> MvsCapture:
        """从已启动的连续预览中读取一帧，不保存且不触发识别。"""

        if not self._opened or self._camera is None or not self._grabbing:
            raise MvsCameraError("PREVIEW_NOT_STARTED", "MVS连续预览尚未启动。")
        frame = self._sdk.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
        ret = self._camera.MV_CC_GetImageBuffer(frame, int(timeout_ms))
        self._check(ret, "PREVIEW_TIMEOUT", "MVS实时预览取帧超时。")
        try:
            if not frame.pBufAddr:
                raise MvsCameraError("EMPTY_FRAME", "MVS返回空预览帧。")
            image = self._frame_to_bgr(frame)
            frame_number = int(frame.stFrameInfo.nFrameNum)
        finally:
            self._check(self._camera.MV_CC_FreeImageBuffer(frame), "FREE_BUFFER_FAILED", "释放 MVS预览缓冲区失败。")
        return MvsCapture(image, frame_number, True, {"trigger_mode": "continuous_preview"})

    def stop_preview(self) -> None:
        if self._camera is not None and self._grabbing:
            self._check(self._camera.MV_CC_StopGrabbing(), "STOP_FAILED", "停止 MVS连续预览失败。")
            self._grabbing = False

    def read_current_parameters(self) -> dict[str, Any]:
        """只读当前采集节点；不启动取流、不触发、不写入参数值。"""

        if not self._opened or self._camera is None:
            raise MvsCameraError("NOT_OPEN", "MVS相机尚未打开。")
        return self._profile_readback()

    def close(self) -> None:
        self._cleanup_camera()
        if self._initialized and self._sdk_exports("MV_CC_Finalize"):
            self._check(self._sdk.MvCamera.MV_CC_Finalize(), "SDK_FINALIZE_FAILED", "MVS SDK反初始化失败。")
            self._initialized = False
        self._release_loader()

    def _sdk_exports(self, name: str) -> bool:
        """检查实际 DLL导出，避免新 wrapper 在旧 Runtime上仅因静态方法存在而误判。"""

        dll = getattr(self._sdk, "MvCamCtrldll", None)
        if dll is None:
            return hasattr(self._sdk.MvCamera, name)
        try:
            getattr(dll, name)
        except AttributeError:
            return False
        return hasattr(self._sdk.MvCamera, name)

    def _apply_profile(self, profile: dict[str, Any], *, require_approved: bool = True) -> None:
        if (require_approved and profile.get("approved") is not True) or profile.get("trigger_mode") != "software":
            raise MvsCameraError("PROFILE_NOT_APPROVED", "相机 profile尚未批准或不是软件触发。")
        exposure = self._finite_positive(profile.get("exposure_us"), "exposure_us")
        gain = self._finite_nonnegative(profile.get("gain"), "gain")
        roi = profile.get("roi")
        white = profile.get("white_balance")
        if not isinstance(roi, dict) or set(roi) != {"width", "height", "offset_x", "offset_y"}:
            raise MvsCameraError("PROFILE_INVALID", "相机 ROI字段无效。")
        if not isinstance(white, dict) or set(white) != {"red", "green", "blue"}:
            raise MvsCameraError("PROFILE_INVALID", "白平衡字段无效。")
        values = {key: self._integer(roi[key], f"roi.{key}") for key in roi}
        if values["width"] <= 0 or values["height"] <= 0 or values["offset_x"] < 0 or values["offset_y"] < 0:
            raise MvsCameraError("PROFILE_INVALID", "相机 ROI数值无效。")
        self._ensure_software_trigger()
        self._set_enum("ExposureAuto", "Off")
        self._set_float("ExposureTime", exposure)
        self._set_enum("GainAuto", "Off")
        self._set_float("Gain", gain)
        self._set_int("OffsetX", 0); self._set_int("OffsetY", 0)
        self._set_int("Width", values["width"]); self._set_int("Height", values["height"])
        self._set_int("OffsetX", values["offset_x"]); self._set_int("OffsetY", values["offset_y"])
        self._set_enum("BalanceWhiteAuto", "Off")
        for selector, key in (("Red", "red"), ("Green", "green"), ("Blue", "blue")):
            self._set_enum("BalanceRatioSelector", selector)
            self._set_balance_ratio(self._finite_positive(white[key], f"white_balance.{key}"))
    @staticmethod
    def _configured_parameters(profile: dict[str, Any]) -> dict[str, Any]:
        """Return the requested values without reading any MVS feature node."""

        return {
            "exposure_us": float(profile["exposure_us"]),
            "gain": float(profile["gain"]),
            "white_balance": {
                key: float(profile["white_balance"][key])
                for key in ("red", "green", "blue")
            },
            "roi": {
                key: int(profile["roi"][key])
                for key in ("width", "height", "offset_x", "offset_y")
            },
            "trigger_mode": "software",
        }

    def _profile_readback(self) -> dict[str, Any]:
        white: dict[str, float] = {}
        for selector, key in (("Red", "red"), ("Green", "green"), ("Blue", "blue")):
            self._set_enum("BalanceRatioSelector", selector)
            white[key] = self._get_balance_ratio()
        return {
            "exposure_us": self._get_float("ExposureTime"),
            "gain": self._get_float("Gain"),
            "white_balance": white,
            "roi": {key: self._get_int(node) for key, node in (("width", "Width"), ("height", "Height"), ("offset_x", "OffsetX"), ("offset_y", "OffsetY"))},
            "trigger_mode": "software",
        }

    def _frame_to_bgr(self, frame: Any) -> np.ndarray:
        info = frame.stFrameInfo
        width, height, length = int(info.nWidth), int(info.nHeight), int(info.nFrameLen)
        if min(width, height, length) <= 0:
            raise MvsCameraError("INVALID_FRAME", "MVS帧尺寸无效。")
        bgr8 = int(self._sdk.PixelType_Gvsp_BGR8_Packed)
        rgb8 = int(self._sdk.PixelType_Gvsp_RGB8_Packed)
        expected = width * height * 3
        if int(info.enPixelType) in (bgr8, rgb8):
            if length < expected:
                raise MvsCameraError("INVALID_FRAME", "MVS彩色帧长度不足。")
            image = np.frombuffer(ctypes.string_at(frame.pBufAddr, expected), dtype=np.uint8).copy().reshape(height, width, 3)
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if int(info.enPixelType) == rgb8 else image)
        destination = (ctypes.c_ubyte * expected)()
        if self._sdk_exports("MV_CC_ConvertPixelTypeEx") and hasattr(self._sdk, "MV_CC_PIXEL_CONVERT_PARAM_EX"):
            parameter = self._sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
            convert = self._camera.MV_CC_ConvertPixelTypeEx
        elif self._sdk_exports("MV_CC_ConvertPixelType") and hasattr(self._sdk, "MV_CC_PIXEL_CONVERT_PARAM"):
            if width > 0xFFFF or height > 0xFFFF:
                raise MvsCameraError("CONVERT_FAILED", "旧版 MVS像素转换接口不支持当前图像尺寸。")
            parameter = self._sdk.MV_CC_PIXEL_CONVERT_PARAM()
            convert = self._camera.MV_CC_ConvertPixelType
        else:
            raise MvsCameraError("CONVERT_FAILED", "当前 MVS Runtime没有可用的像素格式转换接口。")
        ctypes.memset(ctypes.byref(parameter), 0, ctypes.sizeof(parameter))
        parameter.nWidth = width; parameter.nHeight = height
        parameter.pSrcData = frame.pBufAddr; parameter.nSrcDataLen = length
        parameter.enSrcPixelType = int(info.enPixelType); parameter.enDstPixelType = rgb8
        parameter.pDstBuffer = destination; parameter.nDstBufferSize = expected
        self._check(convert(parameter), "CONVERT_FAILED", "MVS像素格式转 BGR失败。")
        if int(parameter.nDstLen) != expected:
            raise MvsCameraError("INVALID_FRAME", "MVS像素转换长度不一致。")
        rgb = np.ctypeslib.as_array(destination).copy().reshape(height, width, 3)
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _describe(self, index: int, info: Any) -> MvsDevice:
        layer = int(info.nTLayerType)
        options = (("MV_GIGE_DEVICE", "GigE", "stGigEInfo"), ("MV_GENTL_GIGE_DEVICE", "GenTL-GigE", "stGigEInfo"), ("MV_USB_DEVICE", "USB3", "stUsb3VInfo"), ("MV_GENTL_CAMERALINK_DEVICE", "CameraLink", "stCMLInfo"), ("MV_GENTL_CXP_DEVICE", "CXP", "stCXPInfo"), ("MV_GENTL_XOF_DEVICE", "XoF", "stXoFInfo"))
        transport, field = f"0x{layer:08X}", ""
        for constant, label, candidate in options:
            if layer == int(getattr(self._sdk, constant, -1)):
                transport, field = label, candidate; break
        special = getattr(info.SpecialInfo, field, None)
        return MvsDevice(index, _decode(getattr(special, "chModelName", b"")), transport)

    def _device_info(self, index: int) -> Any:
        assert self._device_list is not None
        return ctypes.cast(self._device_list.pDeviceInfo[index], ctypes.POINTER(self._sdk.MV_CC_DEVICE_INFO)).contents

    def _set_enum(self, node: str, value: str) -> None:
        self._check(self._camera.MV_CC_SetEnumValueByString(node, value), "PROFILE_SET_FAILED", f"设置 MVS节点 {node}失败。")

    def _ensure_software_trigger(self) -> None:
        """Configure immutable trigger nodes once for each exclusive camera handle."""

        if getattr(self, "_software_trigger_configured", False):
            return
        self._set_enum_idempotent("TriggerMode", "On", 1)
        self._set_enum_idempotent("TriggerSource", "Software", 7)
        self._software_trigger_configured = True

    def _set_enum_idempotent(self, node: str, value: str, expected_numeric: int) -> None:
        try:
            self._set_enum(node, value)
        except MvsCameraError as set_error:
            try:
                current = self._get_enum(node)
            except MvsCameraError:
                raise set_error
            if current != expected_numeric:
                raise set_error

    def _set_float(self, node: str, value: float) -> None:
        self._check(self._camera.MV_CC_SetFloatValue(node, float(value)), "PROFILE_SET_FAILED", f"设置 MVS节点 {node}失败。")

    def _set_int(self, node: str, value: int) -> None:
        self._check(self._camera.MV_CC_SetIntValueEx(node, int(value)), "PROFILE_SET_FAILED", f"设置 MVS节点 {node}失败。")

    def _get_float(self, node: str) -> float:
        value = self._sdk.MVCC_FLOATVALUE(); self._check(self._camera.MV_CC_GetFloatValue(node, value), "PROFILE_READ_FAILED", f"读取 MVS节点 {node}失败。")
        return float(value.fCurValue)

    def _get_enum(self, node: str) -> int:
        value = self._sdk.MVCC_ENUMVALUE(); self._check(self._camera.MV_CC_GetEnumValue(node, value), "PROFILE_READ_FAILED", f"读取 MVS节点 {node}失败。")
        return int(value.nCurValue)

    def _get_balance_ratio(self) -> float:
        """兼容将 BalanceRatio 暴露为 Float 或 Integer 的MVS机型。"""

        try:
            return self._get_float("BalanceRatio")
        except MvsCameraError as float_error:
            try:
                return float(self._get_int("BalanceRatio"))
            except MvsCameraError as int_error:
                raise MvsCameraError(
                    "PROFILE_READ_FAILED",
                    "读取 MVS节点 BalanceRatio失败：既不能按浮点也不能按整数读取。",
                    int_error.sdk_code if int_error.sdk_code is not None else float_error.sdk_code,
                ) from int_error

    def _set_balance_ratio(self, value: float) -> None:
        try:
            self._set_float("BalanceRatio", value)
        except MvsCameraError as float_error:
            if not float(value).is_integer():
                raise MvsCameraError(
                    "PROFILE_INVALID", "当前MVS相机的 BalanceRatio 是整数节点，白平衡必须填整数。"
                ) from float_error
            self._set_int("BalanceRatio", int(value))

    def _get_int(self, node: str) -> int:
        value = self._sdk.MVCC_INTVALUE_EX(); self._check(self._camera.MV_CC_GetIntValueEx(node, value), "PROFILE_READ_FAILED", f"读取 MVS节点 {node}失败。")
        return int(value.nCurValue)

    @staticmethod
    def _finite_positive(value: Any, name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise MvsCameraError("PROFILE_INVALID", f"{name}必须是正有限数。")
        return number

    @staticmethod
    def _finite_nonnegative(value: Any, name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise MvsCameraError("PROFILE_INVALID", f"{name}必须是非负有限数。")
        return number

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MvsCameraError("PROFILE_INVALID", f"{name}必须是整数。")
        return value

    def _cleanup_camera(self) -> None:
        if self._camera is not None:
            if self._grabbing:
                self._camera.MV_CC_StopGrabbing()
            if self._opened:
                self._camera.MV_CC_CloseDevice()
            self._camera.MV_CC_DestroyHandle()
        self._camera = None; self._opened = False; self._grabbing = False
        self._software_trigger_configured = False

    def _release_loader(self) -> None:
        if self._dll_handle is not None:
            self._dll_handle.close(); self._dll_handle = None
        if self._injected_wrapper and str(MVS_WRAPPER_DIR) in sys.path:
            sys.path.remove(str(MVS_WRAPPER_DIR)); self._injected_wrapper = False
        if self._path_before is not None:
            os.environ["PATH"] = self._path_before; self._path_before = None

    def _check(self, ret: object, code: str, message: str) -> None:
        converted = _unsigned(ret)
        if converted != _unsigned(getattr(self._sdk, "MV_OK", 0)):
            raise MvsCameraError(code, f"{message}（MVS SDK返回 0x{converted:08X}）", converted)
