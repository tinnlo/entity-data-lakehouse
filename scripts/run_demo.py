from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from entity_data_lakehouse import run_pipeline

    results = run_pipeline(REPO_ROOT)
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
