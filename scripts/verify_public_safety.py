from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entity_data_lakehouse.public_safety import scan_public_safety


def main() -> None:
    findings = scan_public_safety(REPO_ROOT)
    if findings:
        for finding in findings:
            print(finding)
        raise SystemExit(1)
    print("Public-safety scan passed.")


if __name__ == "__main__":
    main()
