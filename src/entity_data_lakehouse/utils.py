from __future__ import annotations

import hashlib
import re
import unicodedata


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str) -> str:
    """Normalize names for deterministic matching across public source variants."""
    if not value:
        return ""
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    collapsed = _NON_ALNUM.sub(" ", ascii_value).strip()
    return re.sub(r"\s+", " ", collapsed)


def stable_id(prefix: str, *parts: object) -> str:
    material = "||".join(str(part or "") for part in parts)
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
