from __future__ import annotations

import unittest

from app.execution_gate import ExecutionGateError, TwoPhaseExecutionGate
from app.session import CompetitionSession, SessionState


class ExecutionGateTest(unittest.TestCase):
    def recognition(self):
        return {"success": True, "task_type": "task_2", "request_id": "r1", "sequence": []}

    def test_valid_token_is_single_use(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        gate = TwoPhaseExecutionGate(session)
        ready = gate.prepare(self.recognition(), "fingerprint")
        result = gate.commit(request_id="r1", execute_token=ready["execute_token"], config_fingerprint="fingerprint")
        self.assertEqual(result["request_id"], "r1")
        with self.assertRaises(ExecutionGateError):
            gate.commit(request_id="r1", execute_token=ready["execute_token"], config_fingerprint="fingerprint")

    def test_wrong_token_revokes(self) -> None:
        session = CompetitionSession(); session.authorize(True)
        gate = TwoPhaseExecutionGate(session); gate.prepare(self.recognition(), "fingerprint")
        with self.assertRaises(ExecutionGateError):
            gate.commit(request_id="r1", execute_token="wrong", config_fingerprint="fingerprint")
        self.assertEqual(session.state, SessionState.STOPPED)


if __name__ == "__main__":
    unittest.main()
