from __future__ import annotations

import ctypes
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np

from vision.mvs_camera import MvsCamera, MvsCameraError
from pathlib import Path

from vision.contracts import CameraContractError, CaptureRequest, CapturedFrame
from voice.trigger import command_matches


class CameraAndTriggerTest(unittest.TestCase):
    def test_legacy_mvs_runtime_without_global_lifecycle_is_supported(self) -> None:
        class LegacyCamera:
            pass

        camera = MvsCamera(sdk=SimpleNamespace(MvCamera=LegacyCamera, MvCamCtrldll=SimpleNamespace()))
        self.assertFalse(camera._initialized)
        camera.close()

    def test_legacy_pixel_conversion_is_used_for_bayer_frame(self) -> None:
        class LegacyConvertParameter(ctypes.Structure):
            _fields_ = [
                ("nWidth", ctypes.c_ushort), ("nHeight", ctypes.c_ushort),
                ("enSrcPixelType", ctypes.c_int), ("pSrcData", ctypes.POINTER(ctypes.c_ubyte)),
                ("nSrcDataLen", ctypes.c_uint), ("enDstPixelType", ctypes.c_int),
                ("pDstBuffer", ctypes.POINTER(ctypes.c_ubyte)), ("nDstLen", ctypes.c_uint),
                ("nDstBufferSize", ctypes.c_uint), ("nRes", ctypes.c_uint * 4),
            ]

        class LegacyCamera:
            MV_CC_ConvertPixelType = object()

        converted = {"legacy": False}

        class Handle:
            def MV_CC_ConvertPixelType(self, parameter):
                converted["legacy"] = True
                for index in range(int(parameter.nDstBufferSize)):
                    parameter.pDstBuffer[index] = index
                parameter.nDstLen = parameter.nDstBufferSize
                return 0

        source = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        frame = SimpleNamespace(
            pBufAddr=ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)),
            stFrameInfo=SimpleNamespace(nWidth=2, nHeight=2, nFrameLen=4, enPixelType=99),
        )
        camera = MvsCamera.__new__(MvsCamera)
        camera._sdk = SimpleNamespace(
            MvCamera=LegacyCamera,
            MvCamCtrldll=SimpleNamespace(MV_CC_ConvertPixelType=object()),
            MV_CC_PIXEL_CONVERT_PARAM=LegacyConvertParameter,
            PixelType_Gvsp_BGR8_Packed=10,
            PixelType_Gvsp_RGB8_Packed=11,
            MV_OK=0,
        )
        camera._camera = Handle()
        image = camera._frame_to_bgr(frame)
        self.assertTrue(converted["legacy"])
        self.assertEqual(image.shape, (2, 2, 3))

    def test_current_mvs_parameter_read_is_readback_only(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._opened = True
        camera._camera = object()
        camera._profile_readback = Mock(return_value={"exposure_us": 120000.0})
        self.assertEqual(camera.read_current_parameters(), {"exposure_us": 120000.0})
        camera._profile_readback.assert_called_once_with()

    def test_integer_balance_ratio_node_is_supported(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._get_float = Mock(side_effect=MvsCameraError("PROFILE_READ_FAILED", "not a float node"))
        camera._get_int = Mock(return_value=2097)
        self.assertEqual(camera._get_balance_ratio(), 2097.0)

    def test_runtime_profile_apply_never_reads_mvs_parameter_nodes(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._software_trigger_configured = False
        camera._set_enum = Mock()
        camera._set_float = Mock()
        camera._set_int = Mock()
        camera._set_balance_ratio = Mock()
        camera._get_float = Mock(side_effect=AssertionError("runtime must not read float nodes"))
        camera._get_int = Mock(side_effect=AssertionError("runtime must not read integer nodes"))
        camera._get_balance_ratio = Mock(side_effect=AssertionError("runtime must not read balance nodes"))
        camera._apply_profile({
            "approved": True,
            "trigger_mode": "software",
            "exposure_us": 40000,
            "gain": 0,
            "white_balance": {"red": 2097, "green": 1024, "blue": 1559},
            "roi": {"width": 3072, "height": 2048, "offset_x": 0, "offset_y": 0},
        })
        camera._get_float.assert_not_called()
        camera._get_int.assert_not_called()
        camera._get_balance_ratio.assert_not_called()

    def test_software_trigger_nodes_are_only_written_once_per_open_handle(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._software_trigger_configured = False
        camera._set_enum = Mock()

        camera._ensure_software_trigger()
        camera._ensure_software_trigger()

        self.assertEqual(camera._set_enum.call_args_list, [
            call("TriggerMode", "On"),
            call("TriggerSource", "Software"),
        ])

    def test_rejected_redundant_trigger_write_is_accepted_only_after_readback(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._set_enum = Mock(side_effect=MvsCameraError("PROFILE_SET_FAILED", "rejected", 0x80000106))
        camera._get_enum = Mock(return_value=1)

        camera._set_enum_idempotent("TriggerMode", "On", 1)

        camera._get_enum.assert_called_once_with("TriggerMode")

    def test_rejected_trigger_write_with_wrong_readback_still_fails_closed(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._set_enum = Mock(side_effect=MvsCameraError("PROFILE_SET_FAILED", "rejected", 0x80000106))
        camera._get_enum = Mock(return_value=0)

        with self.assertRaises(MvsCameraError):
            camera._set_enum_idempotent("TriggerMode", "On", 1)

    def test_each_software_capture_returns_camera_to_idle(self) -> None:
        class FrameInfo(ctypes.Structure):
            _fields_ = [("nFrameNum", ctypes.c_uint)]

        class Frame(ctypes.Structure):
            _fields_ = [("pBufAddr", ctypes.c_void_p), ("stFrameInfo", FrameInfo)]

        class Handle:
            def __init__(self) -> None:
                self.starts = 0
                self.stops = 0

            def MV_CC_StartGrabbing(self):
                self.starts += 1
                return 0

            def MV_CC_StopGrabbing(self):
                self.stops += 1
                return 0

            def MV_CC_SetCommandValue(self, _name):
                return 0

            def MV_CC_GetImageBuffer(self, frame, _timeout):
                frame.pBufAddr = 1
                frame.stFrameInfo.nFrameNum = self.starts
                return 0

            def MV_CC_FreeImageBuffer(self, _frame):
                return 0

        profile = {
            "approved": True, "trigger_mode": "software", "exposure_us": 40000.0, "gain": 0.0,
            "white_balance": {"red": 2097.0, "green": 1024.0, "blue": 1559.0},
            "roi": {"width": 16, "height": 12, "offset_x": 0, "offset_y": 0},
        }
        handle = Handle()
        camera = MvsCamera.__new__(MvsCamera)
        camera._opened = True
        camera._grabbing = False
        camera._camera = handle
        camera._sdk = SimpleNamespace(MV_FRAME_OUT=Frame, MV_OK=0)
        camera._apply_profile = Mock()
        camera._frame_to_bgr = Mock(return_value=np.zeros((12, 16, 3), dtype=np.uint8))

        first = camera.capture(profile)
        second = camera.capture(profile)

        self.assertEqual((first.frame_number, second.frame_number), (1, 2))
        self.assertEqual((handle.starts, handle.stops), (2, 2))
        self.assertFalse(camera._grabbing)

    def test_single_capture_discards_invalid_frames_and_retriggers(self) -> None:
        class FrameInfo(ctypes.Structure):
            _fields_ = [("nFrameNum", ctypes.c_uint)]

        class Frame(ctypes.Structure):
            _fields_ = [("pBufAddr", ctypes.c_void_p), ("stFrameInfo", FrameInfo)]

        class Handle:
            def __init__(self) -> None:
                self.triggers = 0
                self.frees = 0
                self.stops = 0

            def MV_CC_StartGrabbing(self): return 0
            def MV_CC_StopGrabbing(self): self.stops += 1; return 0
            def MV_CC_SetCommandValue(self, _name): self.triggers += 1; return 0
            def MV_CC_GetImageBuffer(self, frame, _timeout):
                frame.pBufAddr = 1
                frame.stFrameInfo.nFrameNum = self.triggers
                return 0
            def MV_CC_FreeImageBuffer(self, _frame): self.frees += 1; return 0

        profile = {
            "approved": True, "trigger_mode": "software", "exposure_us": 40000.0, "gain": 0.0,
            "white_balance": {"red": 2097.0, "green": 1024.0, "blue": 1559.0},
            "roi": {"width": 16, "height": 12, "offset_x": 0, "offset_y": 0},
        }
        handle = Handle()
        camera = MvsCamera.__new__(MvsCamera)
        camera._opened = True; camera._grabbing = False; camera._camera = handle
        camera._sdk = SimpleNamespace(MV_FRAME_OUT=Frame, MV_OK=0)
        camera._apply_profile = Mock()
        camera._frame_to_bgr = Mock(side_effect=[
            MvsCameraError("INVALID_FRAME", "bad dimensions"),
            MvsCameraError("INVALID_FRAME", "bad dimensions"),
            np.zeros((12, 16, 3), dtype=np.uint8),
        ])

        capture = camera.capture(profile)

        self.assertEqual(capture.frame_number, 3)
        self.assertEqual((handle.triggers, handle.frees, handle.stops), (3, 3, 1))
        self.assertFalse(camera._grabbing)

    def test_single_capture_stops_after_three_invalid_frames(self) -> None:
        class FrameInfo(ctypes.Structure):
            _fields_ = [("nFrameNum", ctypes.c_uint)]

        class Frame(ctypes.Structure):
            _fields_ = [("pBufAddr", ctypes.c_void_p), ("stFrameInfo", FrameInfo)]

        handle = Mock()
        handle.MV_CC_StartGrabbing.return_value = 0
        handle.MV_CC_StopGrabbing.return_value = 0
        handle.MV_CC_SetCommandValue.return_value = 0
        handle.MV_CC_GetImageBuffer.side_effect = lambda frame, _timeout: setattr(frame, "pBufAddr", 1) or 0
        handle.MV_CC_FreeImageBuffer.return_value = 0
        camera = MvsCamera.__new__(MvsCamera)
        camera._opened = True; camera._grabbing = False; camera._camera = handle
        camera._sdk = SimpleNamespace(MV_FRAME_OUT=Frame, MV_OK=0)
        camera._apply_profile = Mock()
        camera._frame_to_bgr = Mock(side_effect=MvsCameraError("INVALID_FRAME", "bad dimensions"))

        with self.assertRaisesRegex(MvsCameraError, "bad dimensions"):
            camera.capture({"approved": True})

        self.assertEqual(handle.MV_CC_SetCommandValue.call_count, 3)
        self.assertEqual(handle.MV_CC_FreeImageBuffer.call_count, 3)
        handle.MV_CC_StopGrabbing.assert_called_once_with()
        self.assertFalse(camera._grabbing)

    def test_continuous_preview_starts_reads_and_stops_without_software_trigger(self) -> None:
        class FrameInfo(ctypes.Structure):
            _fields_ = [("nFrameNum", ctypes.c_uint)]

        class Frame(ctypes.Structure):
            _fields_ = [("pBufAddr", ctypes.c_void_p), ("stFrameInfo", FrameInfo)]

        class Handle:
            def __init__(self) -> None:
                self.starts = 0; self.stops = 0
            def MV_CC_StartGrabbing(self): self.starts += 1; return 0
            def MV_CC_StopGrabbing(self): self.stops += 1; return 0
            def MV_CC_GetImageBuffer(self, frame, _timeout):
                frame.pBufAddr = 1; frame.stFrameInfo.nFrameNum = 42; return 0
            def MV_CC_FreeImageBuffer(self, _frame): return 0

        handle = Handle()
        camera = MvsCamera.__new__(MvsCamera)
        camera._opened = True; camera._grabbing = False; camera._camera = handle
        camera._software_trigger_configured = True
        camera._sdk = SimpleNamespace(MV_FRAME_OUT=Frame, MV_OK=0)
        camera._apply_profile = Mock(); camera._set_enum = Mock()
        camera._frame_to_bgr = Mock(return_value=np.zeros((12, 16, 3), dtype=np.uint8))

        camera.start_preview({"approved": True})
        capture = camera.read_preview_frame()
        camera.stop_preview()

        camera._set_enum.assert_called_once_with("TriggerMode", "Off")
        self.assertEqual(capture.frame_number, 42)
        self.assertEqual(capture.configured_parameters["trigger_mode"], "continuous_preview")
        self.assertEqual((handle.starts, handle.stops), (1, 1))
        self.assertFalse(camera._grabbing)

    def test_single_character_trigger_is_preserved(self) -> None:
        self.assertTrue(command_matches("请"))
        self.assertTrue(command_matches("任务"))
        self.assertFalse(command_matches("完全无关"))

    def test_capture_must_match_serial_request_profile_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp).resolve(); image = session / "task_card" / "one.png"
            image.parent.mkdir(); image.write_bytes(b"capture")
            request = CaptureRequest("r1", session, "task_card", "SERIAL-1")
            frame = CapturedFrame("r1", "SERIAL-1", "task_card", image, "2026-08-10T00:00:00+08:00", True)
            frame.validate_for(request)
            bad = CapturedFrame("r1", "SERIAL-2", "task_card", image, frame.captured_at, True)
            with self.assertRaises(CameraContractError):
                bad.validate_for(request)


if __name__ == "__main__":
    unittest.main()
