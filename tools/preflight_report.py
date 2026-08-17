from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from app.preflight import competition_ready, run_static_preflight


def main() -> int:
    checks = run_static_preflight()
    print(json.dumps({
        "competition_ready": competition_ready(checks),
        "checks": [item.__dict__ for item in checks],
    }, ensure_ascii=False, indent=2))
    return 0 if competition_ready(checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
