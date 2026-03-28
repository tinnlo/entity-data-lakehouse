from __future__ import annotations

from pathlib import Path

from entity_data_lakehouse import run_pipeline


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = run_pipeline(repo_root)
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
