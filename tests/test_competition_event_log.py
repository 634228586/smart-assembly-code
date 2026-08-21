from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.competition_worker import CompetitionWorker
from app.session import CompetitionSession


class CompetitionEventLogTest(unittest.TestCase):
    def test_progress_is_emitted_and_persisted_as_utf8_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            worker = CompetitionWorker(CompetitionSession(), input_mode="voice")
            worker.event_log_path = Path(temp) / "controller" / "competition.jsonl"
            emitted: list[object] = []
            worker.progress.connect(emitted.append)

            worker._record_progress({
                "phase": "asr_recognized",
                "message": "语音识别文字：请开始识别任务卡",
                "instruction": "请开始识别任务卡",
            })

            record = json.loads(worker.event_log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["phase"], "asr_recognized")
            self.assertEqual(record["instruction"], "请开始识别任务卡")
            self.assertEqual(record["input_mode"], "voice")
            self.assertIn("timestamp", record)
            self.assertEqual(emitted[0]["phase"], "asr_recognized")


if __name__ == "__main__":
    unittest.main()
