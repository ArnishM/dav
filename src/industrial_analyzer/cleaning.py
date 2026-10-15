"""Auditing and cleaning for official AI4I data and separate dirty demos."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import read_ai4i_csv
from .validation import (
    BINARY_COLUMNS,
    CANONICAL_COLUMNS,
    MEASUREMENT_COLUMNS,
    validate_ai4i_data,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Path must be a nonempty string or Path")
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = REPOSITORY_ROOT / result
    result = result.resolve()
    try:
        result.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("Path must stay inside the repository") from exc
    return result


def _write_json(path: str | Path, report: dict) -> None:
    report_path = _repo_path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _native(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _outlier_report(df: pd.DataFrame) -> dict:
    fields = {}
    for column in MEASUREMENT_COLUMNS:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            fields[column] = {"status": "unavailable", "flagged_count": 0, "uids": []}
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            fields[column] = {"status": "unavailable", "flagged_count": 0, "uids": []}
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        spread = q3 - q1
        lower = q1 - 1.5 * spread
        upper = q3 + 1.5 * spread
        mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        uids = []
        if "uid" in df.columns:
            uids = [_native(uid) for uid in df.loc[mask, "uid"].tolist()]
        fields[column] = {
            "status": "flagged_only",
            "method": "IQR fences at 1.5 times the interquartile range",
            "lower_fence": lower,
            "upper_fence": upper,
            "flagged_count": int(mask.sum()),
            "uids": uids,
        }
    return fields


def _audit_official(df: pd.DataFrame) -> dict:
    missing = {str(column): int(count) for column, count in df.isna().sum().items()}
    exact_duplicates = int(df.duplicated().sum())
    duplicate_uid_groups = 0
    duplicate_uid_rows = 0
    if "uid" in df.columns:
        present_uid = df["uid"].notna()
        repeated = df.loc[present_uid, "uid"].duplicated(keep=False)
        duplicate_uid_rows = int(repeated.sum())
        duplicate_uid_groups = int(df.loc[present_uid, "uid"][repeated].nunique())

    errors = []
    try:
        validate_ai4i_data(df)
    except ValueError as exc:
        errors.append(str(exc))
    if exact_duplicates:
        errors.append(f"Found {exact_duplicates} exact duplicate row(s)")

    return {
        "missing_values_by_column": missing,
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_uid_groups": duplicate_uid_groups,
        "duplicate_uid_records": duplicate_uid_rows,
        "validation_errors": errors,
        "outliers": _outlier_report(df),
    }


def clean_official_ai4i_data(
    source_path: str | Path = "data/raw/ai4i2020.csv",
    cleaned_path: str | Path = "data/processed/ai4i_cleaned.csv",
    report_path: str | Path = "outputs/reports/data_quality.json",
) -> pd.DataFrame:
    """Audit official data, then write its unchanged values in canonical form.

    The source is read only. Any unexpected schema, value, or duplicate issue
    is recorded in the report and stops creation of the cleaned CSV.
    """
    try:
        df = read_ai4i_csv(source_path)
    except ValueError as exc:
        report = {
            "schema_version": "1.0",
            "dataset_kind": "official",
            "source_path": str(source_path),
            "status": "failed",
            "validation_errors": [str(exc)],
            "actions": {"corrected_values": 0, "removed_rows": 0, "flagged_outliers": 0},
        }
        _write_json(report_path, report)
        raise

    audit = _audit_official(df)
    flagged = sum(
        item["flagged_count"] for item in audit["outliers"].values()
    )
    report = {
        "schema_version": "1.0",
        "dataset_kind": "official",
        "source_path": str(source_path),
        "cleaned_path": str(cleaned_path),
        "status": "failed" if audit["validation_errors"] else "success",
        "row_counts": {
            "input": int(len(df)),
            "output": None if audit["validation_errors"] else int(len(df)),
            "removed": 0,
        },
        "missing_values_by_column": audit["missing_values_by_column"],
        "duplicates": {
            "exact_duplicate_rows": audit["exact_duplicate_rows"],
            "duplicate_uid_groups": audit["duplicate_uid_groups"],
            "duplicate_uid_records": audit["duplicate_uid_records"],
        },
        "validation_errors": audit["validation_errors"],
        "outliers": {
            "method": "IQR fences at 1.5 times the interquartile range",
            "fields": audit["outliers"],
        },
        "actions": {
            "corrected_values": 0,
            "removed_rows": 0,
            "flagged_outliers": flagged,
            "details": [
                "Original sensor values were preserved.",
                "Outliers were flagged and not removed.",
            ],
        },
        "source_preserved": True,
    }

    if audit["validation_errors"]:
        _write_json(report_path, report)
        raise ValueError(
            "Official AI4I data failed validation; see the data-quality report: "
            + "; ".join(audit["validation_errors"])
        )

    cleaned = df.loc[:, CANONICAL_COLUMNS].copy()
    validate_ai4i_data(cleaned)
    output_file = _repo_path(cleaned_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_file, index=False, encoding="utf-8")
    _write_json(report_path, report)
    return cleaned


def _invalid_row_reasons(row: pd.Series) -> list[str]:
    reasons = []
    uid = row["uid"]
    if pd.isna(uid) or isinstance(uid, (bool, np.bool_)):
        reasons.append("uid is missing or not an integer identifier")
    elif not isinstance(uid, (int, float, np.integer, np.floating)):
        reasons.append("uid is missing or not an integer identifier")
    elif not math.isfinite(float(uid)) or not float(uid).is_integer():
        reasons.append("uid is missing or not an integer identifier")

    product_id = row["product_id"]
    if pd.isna(product_id) or not isinstance(product_id, str) or not product_id.strip():
        reasons.append("product_id is missing or invalid")

    category = row["type"]
    if not pd.isna(category) and (not isinstance(category, str) or category not in {"L", "M", "H"}):
        reasons.append("type is not L, M, or H")

    for column in MEASUREMENT_COLUMNS:
        value = row[column]
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            reasons.append(f"{column} is not numeric")
            continue
        if not math.isfinite(float(value)):
            reasons.append(f"{column} is not finite")
            continue
        if column in {"air_temperature_k", "process_temperature_k", "rotational_speed_rpm"} and value <= 0:
            reasons.append(f"{column} must be greater than zero")
        if column in {"torque_nm", "tool_wear_min"} and value < 0:
            reasons.append(f"{column} must be zero or greater")

    for column in BINARY_COLUMNS:
        value = row[column]
        if pd.isna(value):
            reasons.append(f"{column} is missing; labels are never imputed")
        elif isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) or not math.isfinite(float(value)) or float(value) not in {0.0, 1.0}:
            reasons.append(f"{column} must be 0 or 1; labels are never guessed")
    return reasons


def clean_dirty_demo_data(
    df: pd.DataFrame,
    cleaned_path: str | Path = "data/demo/ai4i_dirty_demo_cleaned.csv",
    quarantine_path: str | Path = "data/demo/ai4i_dirty_demo_quarantine.csv",
    report_path: str | Path = "outputs/reports/dirty_demo_quality.json",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean approved dirty-demo cases and quarantine ambiguous records."""
    if df.columns.duplicated().any() or list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError("Dirty demo must have exactly the canonical 14 columns in order")

    work = df.copy().reset_index(drop=True)
    missing_before = {str(col): int(count) for col, count in work.isna().sum().items()}
    exact_duplicate_mask = work.duplicated(keep="first")
    exact_duplicates_removed = int(exact_duplicate_mask.sum())
    work = work.loc[~exact_duplicate_mask].copy().reset_index(drop=True)
    row_numbers = list(range(len(work)))

    reasons_by_row: dict[int, list[str]] = {}
    uid_values = work["uid"]
    present_uids = uid_values.notna()
    repeated_uids = uid_values[present_uids].duplicated(keep=False)
    conflict_uids = set(uid_values[present_uids][repeated_uids].tolist())
    conflict_groups = len(conflict_uids)
    conflict_records = 0

    for position, row in work.iterrows():
        row_reasons = _invalid_row_reasons(row)
        uid = row["uid"]
        if not pd.isna(uid) and uid in conflict_uids:
            row_reasons.append("uid has conflicting records; quarantined")
            conflict_records += 1
        if row_reasons:
            reasons_by_row[position] = row_reasons

    quarantine_positions = set(reasons_by_row)
    clean_positions = [position for position in row_numbers if position not in quarantine_positions]
    clean = work.iloc[clean_positions].copy().reset_index(drop=True)
    quarantine = work.iloc[sorted(quarantine_positions)].copy().reset_index(drop=True)
    if len(quarantine):
        quarantine["quarantine_reason"] = [
            "; ".join(reasons_by_row[position]) for position in sorted(quarantine_positions)
        ]

    corrections: dict[str, int] = {}
    for column in MEASUREMENT_COLUMNS:
        missing_mask = clean[column].isna()
        count = int(missing_mask.sum())
        if count:
            available = pd.to_numeric(clean.loc[~missing_mask, column], errors="coerce")
            median = available.median()
            if pd.isna(median):
                raise ValueError(f"Cannot median-impute {column}: no valid values remain")
            clean.loc[missing_mask, column] = median
            corrections[column] = count

    missing_type = clean["type"].isna()
    if missing_type.any():
        modes = clean.loc[~missing_type, "type"].mode()
        if modes.empty:
            raise ValueError("Cannot mode-impute type: no valid machine types remain")
        clean.loc[missing_type, "type"] = modes.iloc[0]
        corrections["type"] = int(missing_type.sum())

    clean["uid"] = pd.to_numeric(clean["uid"], errors="raise").astype("int64")
    for column in BINARY_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="raise").astype("int64")
    for column in MEASUREMENT_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    clean = clean.loc[:, CANONICAL_COLUMNS]

    validation_error = None
    try:
        validate_ai4i_data(clean)
    except ValueError as exc:
        validation_error = str(exc)

    outliers = _outlier_report(clean)
    flagged = sum(item["flagged_count"] for item in outliers.values())
    report = {
        "schema_version": "1.0",
        "dataset_kind": "dirty_demonstration",
        "status": "failed" if validation_error else "success",
        "row_counts": {
            "input": int(len(df)),
            "after_exact_duplicate_removal": int(len(df) - exact_duplicates_removed),
            "cleaned_output": int(len(clean)),
            "quarantined": int(len(quarantine)),
            "exact_duplicate_rows_removed": exact_duplicates_removed,
        },
        "missing_values_before_by_column": missing_before,
        "missing_values_after_by_column": {
            str(col): int(count) for col, count in clean.isna().sum().items()
        },
        "duplicates": {
            "exact_duplicate_rows_removed": exact_duplicates_removed,
            "conflicting_uid_groups_quarantined": conflict_groups,
            "conflicting_uid_records_quarantined": conflict_records,
        },
        "imputations": corrections,
        "outliers": {
            "method": "IQR fences at 1.5 times the interquartile range",
            "fields": outliers,
        },
        "actions": {
            "corrected_values": int(sum(corrections.values())),
            "removed_exact_duplicate_rows": exact_duplicates_removed,
            "quarantined_rows": int(len(quarantine)),
            "flagged_outliers": flagged,
        },
        "validation_error": validation_error,
    }

    cleaned_file = _repo_path(cleaned_path)
    quarantine_file = _repo_path(quarantine_path)
    quarantine_file.parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_csv(quarantine_file, index=False, encoding="utf-8")
    _write_json(report_path, report)
    if validation_error:
        raise ValueError(f"Cleaned dirty demo failed validation: {validation_error}")
    cleaned_file.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(cleaned_file, index=False, encoding="utf-8")
    return clean, quarantine, report
