from __future__ import annotations

from pathlib import Path


_FA = "F" + "A"
_FORWARD = "Forward"
_ANALYTICS = "Analytics"
_PIPELINE = "data" + "_" + "pipeline"

BANNED_TOKENS = [
    f"{_FORWARD} {_ANALYTICS}",
    (_FORWARD + _ANALYTICS).casefold(),
    f"{_FA}_{_PIPELINE}",
    "f" + "a" + "_",
]


def scan_public_safety(repo_root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".parquet", ".duckdb", ".pyc"}:
            continue
        text = path.read_text(errors="ignore")
        for token in BANNED_TOKENS:
            if token in text:
                findings.append(f"{path.relative_to(repo_root)} contains banned token '{token}'")
        internal_path_marker = "/Users/lxt/Documents/" + _FORWARD + _ANALYTICS + "/"
        if internal_path_marker in text:
            findings.append(f"{path.relative_to(repo_root)} contains an internal absolute path")
    return findings
