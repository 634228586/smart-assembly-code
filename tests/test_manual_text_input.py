from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock

from app.manual_text_input import CountdownInput, ManualTextInput, ManualTextInputStopped


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

    def test_countdown_mode_sends_wakeup_once_then_command_each_time(self) -> None:
        channel = CountdownInput(wakeup_delay_s=0.0, command_delay_s=0.0)
        events: list[dict[str, object]] = []
        wakeups: list[str] = []
        stop_event = threading.Event()

        first = channel.listen(
            True,
            stop_event=stop_event,
            progress=events.append,
            on_wakeup=lambda: wakeups.append("ready"),
        )
        second = channel.listen(
            False,
            stop_event=stop_event,
            progress=events.append,
            on_wakeup=lambda: wakeups.append("unexpected"),
        )

        self.assertEqual(first, "请开始识别任务卡")
        self.assertEqual(second, "请开始识别任务卡")
        self.assertEqual(wakeups, ["ready"])
        self.assertEqual(sum(event["phase"] == "countdown_wakeup_sent" for event in events), 1)
        self.assertEqual(sum(event["phase"] == "countdown_command_sent" for event in events), 2)

    def test_countdown_mode_is_interruptible(self) -> None:
        stop_event = threading.Event(); stop_event.set()
        with self.assertRaises(ManualTextInputStopped):
            CountdownInput().listen(
                True,
                stop_event=stop_event,
                progress=lambda _event: None,
                on_wakeup=lambda: None,
            )

    def test_second_task_card_uses_separate_twelve_second_delay(self) -> None:
        channel = CountdownInput(
            wakeup_delay_s=5.0,
            command_delay_s=5.0,
            next_command_delay_s=12.0,
        )
        countdown = Mock()
        channel._countdown = countdown  # type: ignore[method-assign]
        stop_event = threading.Event()

        channel.listen(True, stop_event=stop_event, progress=lambda _event: None, on_wakeup=lambda: None)
        channel.listen(False, stop_event=stop_event, progress=lambda _event: None, on_wakeup=lambda: None)

        self.assertEqual([call.args[0] for call in countdown.call_args_list], [5.0, 5.0, 12.0])


if __name__ == "__main__":
    unittest.main()
