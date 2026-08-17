from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FormalSourceAuditTest(unittest.TestCase):
    def test_runtime_has_no_development_fallback_markers(self) -> None:
        forbidden = (
            "mock", "fixed_test", "synthetic", "sim_test", "Aubo Sim",
            "--simulate-speech", "E:\\iwen-codex", "192.168.226.130", "123456",
        )
        violations = []
        for directory in (ROOT / "app", ROOT / "vision", ROOT / "voice"):
            if not directory.exists():
                continue
            for path in directory.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for marker in forbidden:
                    if marker in text:
                        violations.append(f"{path.relative_to(ROOT)}: {marker}")
        self.assertEqual(violations, [])

    def test_no_historical_numeric_step_imports(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app").rglob("*.py"))
        self.assertNotIn("qt_learning", text)
        self.assertNotIn("spec_from_file_location", text)

    def test_launchers_and_tools_have_no_development_python_fallback(self) -> None:
        violations = []
        for path in list(ROOT.glob("*.ps1")) + list(ROOT.glob("*.cmd")) + list((ROOT / "tools").glob("*.ps1")):
            text = path.read_text(encoding="utf-8")
            if "QT MAKING" in text or "\\.venv\\Scripts\\python.exe" in text:
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_detector_approval_is_not_a_runtime_blocker(self) -> None:
        direct_worker = (ROOT / "app" / "direct_assembly_worker.py").read_text(encoding="utf-8")
        localizer = (ROOT / "vision" / "workspace_localizer.py").read_text(encoding="utf-8")
        self.assertNotIn('detector.get("approved") is not True', direct_worker)
        locate_body = localizer.split("def locate_colors", 1)[1].split("def detect_color_pixel", 1)[0]
        self.assertNotIn('detector.get("approved")', locate_body)


if __name__ == "__main__":
    unittest.main()
