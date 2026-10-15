"""Validated descriptive features for canonical AI4I observations."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "uid", "product_id", "type", "air_temperature_k",
    "process_temperature_k", "rotational_speed_rpm", "torque_nm",
    "tool_wear_min", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf",
)
DERIVED_COLUMNS = ("temperature_difference_k", "torque_speed_product")
SENSOR_COLUMNS = CANONICAL_COLUMNS[3:8]
LABEL_COLUMNS = CANONICAL_COLUMNS[8:]


def _validate_canonical(df: pd.DataFrame) -> None:
    """Validate this module's input without cleaning or changing it."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df must contain at least one row")
    if not df.columns.is_unique:
        raise ValueError("df contains duplicate column names")
    collisions = [name for name in DERIVED_COLUMNS if name in df.columns]
    if collisions:
        raise ValueError(f"Derived-column name collision: {collisions}")
    if list(df.columns) != list(CANONICAL_COLUMNS):
        missing = [name for name in CANONICAL_COLUMNS if name not in df.columns]
        extra = [name for name in df.columns if name not in CANONICAL_COLUMNS]
        raise ValueError(
            "Expected exactly the 14 canonical columns in contract order; "
            f"missing={missing}, unexpected={extra}"
        )
    if df.isna().any().any():
        names = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values in columns: {names}")

    for name in ("uid", *LABEL_COLUMNS):
        for value in df[name]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must contain integers, not booleans or floats")
            if name != "uid" and value not in (0, 1):
                raise ValueError(f"{name} must contain binary integers 0 or 1")
    if not df["uid"].is_unique:
        raise ValueError("uid must contain unique integer identifiers")
    if not df["product_id"].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("product_id must contain nonempty strings")
    if not df["type"].isin(("L", "M", "H")).all():
        raise ValueError("type must be one of L, M, H")

    for name in SENSOR_COLUMNS:
        for value in df[name]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must contain real numbers, not strings or booleans")
            try:
                finite = np.isfinite(float(value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{name} must contain finite numerical values")
            strictly_positive = name in SENSOR_COLUMNS[:3]
            if value < 0 or (strictly_positive and value == 0):
                rule = "greater than zero" if strictly_positive else "zero or greater"
                raise ValueError(f"{name} must be {rule}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new canonical DataFrame with two descriptive features appended.

    Preserve original values, dtypes, index and row/column order. Temperature
    difference is in Kelvin; torque-speed product is in Nm × RPM, not watts.
    Neither derived feature belongs to the initial prediction input allowlist.
    Invalid schema, values or calculated overflow raise descriptive ValueError.
    """
    _validate_canonical(df)
    # Convert operands to float before arithmetic to avoid integer wraparound.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = (
            df["process_temperature_k"].to_numpy(dtype=float)
            - df["air_temperature_k"].to_numpy(dtype=float)
        )
        product = (
            df["torque_nm"].to_numpy(dtype=float)
            * df["rotational_speed_rpm"].to_numpy(dtype=float)
        )
    for name, values in zip(DERIVED_COLUMNS, (difference, product)):
        if not np.isfinite(values).all():
            raise ValueError(f"Calculated {name} contains nonfinite values")
    result = df.copy(deep=True)
    result[DERIVED_COLUMNS[0]] = difference
    result[DERIVED_COLUMNS[1]] = product
    return result
