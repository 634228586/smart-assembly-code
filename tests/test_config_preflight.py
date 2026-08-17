from __future__ import annotations

import unittest

from app.config import ConfigurationError, endpoints, load_all
from app.preflight import competition_ready, run_static_preflight


class ConfigPreflightTest(unittest.TestCase):
    def test_real_configs_load_and_are_fail_closed(self) -> None:
        configs = load_all()
        self.assertEqual(configs["camera"]["mounting"], "eye_in_hand")
        checks = run_static_preflight()
        release_check = next(item for item in checks if item.item == "真实硬件验收放行")
        self.assertFalse(release_check.critical)
        self.assertIn(release_check.status, {"PASS", "WARN"})
        self.assertEqual(
            competition_ready(checks),
            not any(item.status == "FAIL" and item.critical for item in checks),
        )
        self.assertTrue(any(item.item == "包内真实 MVS 服务" and item.status == "PASS" for item in checks))
        self.assertTrue(any(item.item == "包内 MVS Python wrapper" and item.status == "PASS" for item in checks))
        profile_check = next(item for item in checks if item.item == "三套采集参数批准")
        profiles = configs["camera"]["profiles"]
        expected_profile_status = "PASS" if all(
            profiles[name]["approved"] is True for name in ("task_card", "blocks", "trays")
        ) else "FAIL"
        self.assertEqual(profile_check.status, expected_profile_status)
        detector_check = next(
            item for item in checks
            if not item.critical and "'blocks':" in item.actual and "'trays':" in item.actual
        )
        self.assertEqual(detector_check.status, "PASS")
        self.assertFalse(detector_check.critical)
        removed = {
            "robot", "camera", "baseline", "工具安装核验", "负载核验", "真实机器人总放行",
            "批准配置哈希", "批准代码哈希", "批准标定哈希", "真实接线核验", "光圈真实接线核验",
        }
        self.assertFalse(any(item.item in removed for item in checks))

    def test_endpoint_roles_are_not_confused(self) -> None:
        config = load_all()["endpoints"]
        result = endpoints(config)
        self.assertEqual(result["robot_rpc"].host, "192.168.34.10")
        self.assertEqual(result["robot_rpc"].port, 30004)
        self.assertEqual(result["robot_rpc"].expected_service, "aubo_rpc")
        self.assertEqual(result["vision_service"].expected_service, "real_mvs_vision")
        self.assertEqual(result["qt_command_listener"].direction, "listen")
        self.assertEqual(result["speech_service"].host, "192.168.34.200")
        self.assertEqual(result["speech_service"].port, 8765)
        self.assertEqual(result["speech_service"].expected_service, "arm_speech_service")
        self.assertEqual(dict(result["speech_service"].routes), {
            "health": "/health", "asr": "/v1/asr", "tts": "/v1/tts",
        })

    def test_duplicate_listener_is_rejected(self) -> None:
        config = load_all()["endpoints"]
        config["vision_service"]["direction"] = "listen"
        config["vision_service"]["port"] = config["qt_command_listener"]["port"]
        with self.assertRaises(ConfigurationError):
            endpoints(config)

    def test_speech_routes_are_required_and_must_be_local_paths(self) -> None:
        config = load_all()["endpoints"]
        del config["speech_service"]["routes"]["tts"]
        with self.assertRaises(ConfigurationError):
            endpoints(config)

        config = load_all()["endpoints"]
        config["speech_service"]["routes"]["asr"] = "http://other-host/v1/asr"
        with self.assertRaises(ConfigurationError):
            endpoints(config)


if __name__ == "__main__":
    unittest.main()
