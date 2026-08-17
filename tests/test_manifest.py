from __future__ import annotations

import unittest

from tools.verify_manifest import main


class ManifestTest(unittest.TestCase):
    def test_manifest_matches_files(self) -> None:
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
