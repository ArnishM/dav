"""Load the official AI4I CSV into the shared canonical column schema."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .validation import CANONICAL_COLUMNS, validate_ai4i_data


SOURCE_TO_CANONICAL = {
    "UDI": "uid",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_ai4i_csv(path: str | Path) -> pd.DataFrame:
    """Read and rename a raw-source or canonical AI4I CSV.

    Relative paths are resolved from the repository root. The loader never
    downloads data, validates values, or modifies the source file.
    """
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("CSV path must be a nonempty string or Path")

    csv_path = Path(path).expanduser()
    if not csv_path.is_absolute():
        csv_path = _repository_root() / csv_path

    # pandas may silently rename duplicate CSV headers, so inspect the actual
    # first row before loading the table.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {csv_path}") from exc
    if len(header) != len(set(header)):
        raise ValueError("CSV contains duplicate column names")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if list(df.columns) == list(SOURCE_TO_CANONICAL):
        df = df.rename(columns=SOURCE_TO_CANONICAL)
    elif list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            "CSV headers must match the official AI4I source headers or the "
            "canonical schema"
        )

    return df


def load_ai4i_csv(path: str | Path) -> pd.DataFrame:
    """Read an AI4I CSV, map its headers, and validate canonical values."""
    df = read_ai4i_csv(path)
    validate_ai4i_data(df)
    return df
