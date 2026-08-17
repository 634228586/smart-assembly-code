from __future__ import annotations

import threading
import unittest

from app.manual_text_input import ManualTextInput, ManualTextInputStopped


class ManualTextInputTest(unittest.TestCase):
    def test_wakeup_is_required_then_next_text_is_returned(self) -> None:
        channel = ManualTextInput()
        events: list[dict[str, object]] = []
        replies: list[str] = []
        channel.submit("无关内容")
        channel.submit("小具 同学")
        channel.submit("开始执行任务卡")

        instruction = channel.listen(
            True,
            stop_event=threading.Event(),
            progress=events.append,
            on_wakeup=lambda: replies.append("ready"),
        )

        self.assertEqual(instruction, "开始执行任务卡")
        self.assertEqual(replies, ["ready"])
        self.assertTrue(any(event["phase"] == "manual_text_ignored" for event in events))

    def test_stop_interrupts_wait(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        with self.assertRaises(ManualTextInputStopped):
            ManualTextInput().listen(
                False,
                stop_event=stop_event,
                progress=lambda _event: None,
                on_wakeup=lambda: None,
            )


if __name__ == "__main__":
    unittest.main()
