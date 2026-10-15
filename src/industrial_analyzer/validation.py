"""Validation for the shared AI4I canonical data schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


CANONICAL_COLUMNS = [
    "uid",
    "product_id",
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

MEASUREMENT_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

BINARY_COLUMNS = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]


def validate_ai4i_data(df: pd.DataFrame) -> None:
    """Validate a canonical AI4I DataFrame, raising ``ValueError`` on issues.

    The function checks the shared schema and operating-domain rules only. It
    does not clean, reorder, or modify the supplied DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("AI4I data must be provided as a pandas DataFrame")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {duplicates}")
    if df.empty:
        raise ValueError("AI4I data must contain at least one row")
    if list(df.columns) != CANONICAL_COLUMNS:
        missing = [name for name in CANONICAL_COLUMNS if name not in df.columns]
        extra = [name for name in df.columns if name not in CANONICAL_COLUMNS]
        raise ValueError(
            "Columns must exactly match the canonical schema and order; "
            f"missing={missing}, extra={extra}"
        )

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        details = ", ".join(f"{name}: {int(count)}" for name, count in missing_values.items())
        raise ValueError(f"Required values are missing ({details})")

    if not is_integer_dtype(df["uid"].dtype) or is_bool_dtype(df["uid"].dtype):
        raise ValueError("uid must contain integer identifiers")
    if df["uid"].duplicated().any():
        raise ValueError("uid values must be unique")

    for column in ("product_id", "type"):
        if not df[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain strings")
        if not df[column].map(lambda value: bool(value.strip())).all():
            raise ValueError(f"{column} must not contain empty strings")

    if not df["type"].isin(["L", "M", "H"]).all():
        raise ValueError("type values must be exactly 'L', 'M', or 'H'")

    for column in MEASUREMENT_COLUMNS:
        series = df[column]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            raise ValueError(f"{column} must contain numeric values")
        try:
            finite = np.isfinite(series.to_numpy(dtype=float)).all()
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{column} must contain only finite values")

    for column in ("air_temperature_k", "process_temperature_k", "rotational_speed_rpm"):
        if (df[column] <= 0).any():
            raise ValueError(f"{column} values must be greater than zero")
    for column in ("torque_nm", "tool_wear_min"):
        if (df[column] < 0).any():
            raise ValueError(f"{column} values must be zero or greater")

    for column in BINARY_COLUMNS:
        series = df[column]
        if not is_integer_dtype(series.dtype) or is_bool_dtype(series.dtype):
            raise ValueError(f"{column} must contain integer labels 0 or 1")
        if not series.isin([0, 1]).all():
            raise ValueError(f"{column} values must be 0 or 1")
