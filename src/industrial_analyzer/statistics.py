"""Descriptive tables for the canonical AI4I dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import BINARY_COLUMNS, MEASUREMENT_COLUMNS, validate_ai4i_data


def summary_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the five physical sensor measurements in their source units."""
    validate_ai4i_data(df)
    table = df.loc[:, MEASUREMENT_COLUMNS].describe().transpose()
    table.index.name = "feature"
    return table.reset_index()


def failure_by_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Report records and recorded failures for each product type."""
    validate_ai4i_data(df)
    grouped = df.groupby("type", observed=False)["machine_failure"].agg(
        total_records="size", failure_count="sum"
    )
    grouped = grouped.reindex(["L", "M", "H"], fill_value=0)
    table = grouped.rename_axis("type").reset_index()
    table["nonfailure_count"] = table["total_records"] - table["failure_count"]
    table["failure_rate_percent"] = (
        table["failure_count"] / table["total_records"].replace(0, pd.NA) * 100
    )
    return table.loc[
        :, ["type", "total_records", "failure_count", "nonfailure_count", "failure_rate_percent"]
    ]


def failure_modes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize descriptive failure-mode flags; modes can overlap."""
    validate_ai4i_data(df)
    count = len(df)
    return pd.DataFrame(
        {
            "failure_mode": BINARY_COLUMNS[1:],
            "flagged_records": [int(df[column].sum()) for column in BINARY_COLUMNS[1:]],
            "percent_of_records": [
                float(df[column].mean() * 100) for column in BINARY_COLUMNS[1:]
            ],
            "dataset_records": count,
        }
    )


def write_statistics_tables(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/tables",
) -> dict[str, Path]:
    """Write the three agreed descriptive tables without CSV index columns."""
    directory = Path(output_dir).expanduser()
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[2] / directory
    directory.mkdir(parents=True, exist_ok=True)

    tables = {
        "summary_statistics": summary_statistics_table(df),
        "failure_by_type": failure_by_type_table(df),
        "failure_modes": failure_modes_table(df),
    }
    paths = {name: directory / f"{name}.csv" for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False, encoding="utf-8")
    return paths
