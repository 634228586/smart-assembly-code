from __future__ import annotations

import http.client
import json
import unittest
from unittest.mock import patch

from app.config import Endpoint
from voice.speech_client import SpeechError, _request, speak


ENDPOINT = Endpoint(
    "speech_service", "192.168.34.200", 8765, "outbound", "arm_speech_service",
    {"health": "/health", "asr": "/v1/asr", "tts": "/v1/tts"},
)


class SpeechClientTest(unittest.TestCase):
    @patch("voice.speech_client.urllib.request.urlopen")
    def test_tts_transport_retries_one_remote_disconnect(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({"ok": True}).encode("utf-8")

        urlopen.side_effect = [http.client.RemoteDisconnected("closed"), Response()]

        result = _request(
            ENDPOINT, "POST", "/v1/tts", {"text": "测试"}, 5.0,
            retry_remote_disconnect=True,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    @patch("voice.speech_client._request")
    def test_tts_requires_real_matching_spoken_text(self, request) -> None:
        request.return_value = {
            "ok": True, "message": "tts ok",
            "result_digest": {"tts_backend": "piper", "spoken_text": "测试成功"},
        }
        speak(ENDPOINT, "测试成功")

    @patch("voice.speech_client._request")
    def test_tts_uses_requested_timeout_without_transport_retry(self, request) -> None:
        request.return_value = {
            "ok": True, "message": "tts ok",
            "result_digest": {"tts_backend": "piper", "spoken_text": "测试成功"},
        }

        speak(ENDPOINT, "测试成功", timeout_s=30.0, retry_remote_disconnect=False)

        self.assertEqual(request.call_args.args[4], 30.0)
        self.assertFalse(request.call_args.kwargs["retry_remote_disconnect"])

    @patch("voice.speech_client._request")
    def test_tts_skipped_is_not_success(self, request) -> None:
        request.return_value = {
            "ok": True, "message": "tts skipped due to unavailable runtime",
            "result_digest": {"tts_skipped": True, "skip_code": "PIPER_SYNTH_FAILED", "spoken_text": "测试成功"},
        }
        with self.assertRaises(SpeechError):
            speak(ENDPOINT, "测试成功")

    @patch("voice.speech_client._request")
    def test_tts_mismatched_text_is_not_success(self, request) -> None:
        request.return_value = {
            "ok": True, "message": "tts ok",
            "result_digest": {"tts_backend": "piper", "spoken_text": "其他文字"},
        }
        with self.assertRaises(SpeechError):
            speak(ENDPOINT, "测试成功")


if __name__ == "__main__":
    unittest.main()
