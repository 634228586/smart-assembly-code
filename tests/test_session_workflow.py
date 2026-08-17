from __future__ import annotations

import unittest

from app.session import CompetitionSession, SessionState
from app.workflow import Phase, build_phases


COLORS = ["红", "橙", "黄", "绿", "蓝", "紫"]


def task(task_type: str, request_id: str) -> dict:
    value = {"success": True, "task_type": task_type, "request_id": request_id}
    if task_type == "task_2":
        value["sequence"] = [
            {"order": i + 1, "block_color": color, "tray_color": COLORS[(i + 1) % 6]}
            for i, color in enumerate(COLORS)
        ]
    return value


class SessionWorkflowTest(unittest.TestCase):
    def test_task1_then_task2_and_duplicate(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        self.assertEqual(session.accept_recognition(task("task_1", "one")), "task_1_completed")
        self.assertEqual(session.accept_recognition(task("task_1", "repeat")), "duplicate_ignored")
        self.assertEqual(session.accept_recognition(task("task_2", "two")), "task_2_execute")
        for _ in range(6): session.cycle_completed()
        session.task_2_return_completed()
        self.assertEqual(session.state, SessionState.COMPLETE)

    def test_task2_then_task1(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        self.assertEqual(session.accept_recognition(task("task_2", "two")), "task_2_execute")
        for _ in range(6): session.cycle_completed()
        session.task_2_return_completed()
        self.assertEqual(session.state, SessionState.AUTHORIZED)
        self.assertEqual(session.accept_recognition(task("task_1", "one")), "task_1_completed")
        self.assertEqual(session.state, SessionState.COMPLETE)

    def test_fault_revokes_authorization(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        session.revoke("camera disconnected", suction_state="possibly_on")
        self.assertEqual(session.state, SessionState.STOPPED)
        self.assertEqual(session.suction_state, "possibly_on")

    def test_skipped_cycles_count_as_processed_but_not_completed(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        self.assertEqual(session.accept_recognition(task("task_2", "two")), "task_2_execute")
        for _ in range(4): session.cycle_completed()
        for _ in range(2): session.cycle_skipped()
        session.task_2_return_completed()
        self.assertEqual(session.completed_cycles, 4)
        self.assertEqual(session.skipped_cycles, 2)
        self.assertIn("task_2", session.processed_task_types)

    def test_six_cycles_have_78_phases(self) -> None:
        phases = build_phases(task("task_2", "two")["sequence"])
        self.assertEqual(len(phases), 78)
        self.assertEqual(phases[0].phase, Phase.BLOCK_PHOTO_POINT)
        self.assertEqual(phases[-1].phase, Phase.PLACE_UP)


if __name__ == "__main__":
    unittest.main()
