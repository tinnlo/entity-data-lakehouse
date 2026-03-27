from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_TYPE_CHECKS = {
    "string": lambda series: pd.api.types.is_string_dtype(series)
    or pd.api.types.is_object_dtype(series),
    "number": pd.api.types.is_numeric_dtype,
    "integer": pd.api.types.is_integer_dtype,
    "boolean": pd.api.types.is_bool_dtype,
}


def load_contract(contract_path: Path) -> dict:
    return json.loads(contract_path.read_text())


def validate_dataframe(df: pd.DataFrame, contract_path: Path) -> None:
    contract = load_contract(contract_path)
    required = contract.get("required", [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{contract_path.name}: missing required columns {missing}")

    properties = contract.get("properties", {})
    for column, definition in properties.items():
        if column not in df.columns:
            continue
        expected_type = definition.get("type")
        checker = _TYPE_CHECKS.get(expected_type)
        if checker is None:
            continue
        if not checker(df[column]):
            raise ValueError(
                f"{contract_path.name}: column '{column}' is not compatible with type '{expected_type}'"
            )
