"""Create a separate transformed dataset for DAV demonstrations."""

from __future__ import annotations

import pandas as pd

from .validation import CANONICAL_COLUMNS, MEASUREMENT_COLUMNS, validate_ai4i_data


def transform_for_dav(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical fields plus encoded and scaled demonstration columns.

    Scaling and quantile bins are fitted across this descriptive dataset and
    must not be used as model inputs. The original 14 fields are preserved.
    """
    validate_ai4i_data(df)
    transformed = df.loc[:, CANONICAL_COLUMNS].copy(deep=True)

    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype="int64")
    type_dummies = type_dummies.reindex(columns=["type_L", "type_M", "type_H"], fill_value=0)
    for column in type_dummies.columns:
        transformed[column] = type_dummies[column].to_numpy()

    for column in MEASUREMENT_COLUMNS:
        values = df[column].astype(float)
        value_range = values.max() - values.min()
        transformed[f"{column}_minmax"] = (
            (values - values.min()) / value_range if value_range else 0.0
        )
        standard_deviation = values.std(ddof=0)
        transformed[f"{column}_zscore"] = (
            (values - values.mean()) / standard_deviation
            if standard_deviation
            else 0.0
        )

    unique_wear_values = int(df["tool_wear_min"].nunique())
    if unique_wear_values == 1:
        transformed["tool_wear_quantile_bin"] = 1
    else:
        bins = pd.qcut(
            df["tool_wear_min"],
            q=min(4, unique_wear_values),
            labels=False,
            duplicates="drop",
        )
        transformed["tool_wear_quantile_bin"] = bins.astype("int64") + 1
    return transformed


def write_transformed_csv(
    df: pd.DataFrame,
    output_path: str = "data/processed/ai4i_transformed.csv",
) -> pd.DataFrame:
    """Write the demonstration transform without changing the input CSV."""
    from pathlib import Path

    transformed = transform_for_dav(df)
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(path, index=False, encoding="utf-8")
    return transformed
