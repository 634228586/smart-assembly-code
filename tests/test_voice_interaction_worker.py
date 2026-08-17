from __future__ import annotations

import unittest
from unittest.mock import patch

from app.voice_interaction_worker import VoiceInteractionWorker


class VoiceInteractionWorkerTest(unittest.TestCase):
    @patch("app.voice_interaction_worker.speech_client.listen")
    @patch("app.voice_interaction_worker.speech_client.health")
    def test_listen_returns_recognized_text(self, health, listen) -> None:
        health.return_value = {"service": "arm_speech_service", "version": "test"}
        listen.return_value = "请开始识别任务卡"
        worker = VoiceInteractionWorker("listen", wakeup_required=True, timeout_s=12.0)
        results: list[object] = []
        failures: list[str] = []
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)

        worker.run()

        self.assertEqual(failures, [])
        self.assertEqual(results, [{"action": "listen", "recognized_text": "请开始识别任务卡"}])
        listen.assert_called_once()
        self.assertTrue(listen.call_args.kwargs["wakeup_required"])

    @patch("app.voice_interaction_worker.speech_client.speak")
    @patch("app.voice_interaction_worker.speech_client.health")
    def test_speak_uses_entered_text(self, health, speak) -> None:
        health.return_value = {"service": "arm_speech_service", "version": "test"}
        worker = VoiceInteractionWorker("speak", text="  测试播报  ")
        results: list[object] = []
        worker.finished.connect(results.append)

        worker.run()

        self.assertEqual(results, [{"action": "speak", "spoken_text": "测试播报"}])
        self.assertEqual(speak.call_args.args[1], "测试播报")


if __name__ == "__main__":
    unittest.main()
