"""Tests for official-data and dirty-demo cleaning policies."""

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from industrial_analyzer.cleaning import clean_dirty_demo_data, clean_official_ai4i_data
from industrial_analyzer.data_loader import load_ai4i_csv
from industrial_analyzer.validation import CANONICAL_COLUMNS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def output_directory():
    parent = REPOSITORY_ROOT / "outputs" / "reports"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".cleaning-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_row(uid):
    return {
        "uid": uid,
        "product_id": f"M{uid:05d}",
        "type": "M",
        "air_temperature_k": 298.0 + uid,
        "process_temperature_k": 308.0 + uid,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 10,
        "machine_failure": 0,
        "twf": 0,
        "hdf": 0,
        "pwf": 0,
        "osf": 0,
        "rnf": 0,
    }


def test_official_cleaning_preserves_data_and_reports_outliers(output_directory):
    source_path = REPOSITORY_ROOT / "data" / "raw" / "ai4i2020.csv"
    original_bytes = source_path.read_bytes()
    source = load_ai4i_csv(source_path)
    output_path = output_directory / "cleaned.csv"
    report_path = output_directory / "data_quality.json"

    cleaned = clean_official_ai4i_data(source_path, output_path, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reloaded = pd.read_csv(output_path)

    assert cleaned.shape == (10_000, 14)
    assert reloaded.columns.tolist() == CANONICAL_COLUMNS
    for column in (
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ):
        pd.testing.assert_series_equal(source[column], reloaded[column], check_names=True)
    assert source_path.read_bytes() == original_bytes
    assert report["status"] == "success"
    assert sum(report["missing_values_by_column"].values()) == 0
    assert report["actions"]["removed_rows"] == 0
    assert report["actions"]["flagged_outliers"] > 0


def test_official_invalid_data_is_reported_and_stops(output_directory):
    source_path = output_directory / "invalid.csv"
    row = make_row(1)
    row["torque_nm"] = None
    pd.DataFrame([row], columns=CANONICAL_COLUMNS).to_csv(source_path, index=False)
    output_path = output_directory / "must_not_be_written.csv"
    report_path = output_directory / "data_quality.json"

    with pytest.raises(ValueError, match="failed validation"):
        clean_official_ai4i_data(source_path, output_path, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["missing_values_by_column"]["torque_nm"] == 1
    assert not output_path.exists()


def test_dirty_demo_imputes_removes_and_quarantines_as_agreed(output_directory):
    frame = pd.DataFrame([make_row(uid) for uid in range(1, 6)], columns=CANONICAL_COLUMNS)
    frame.loc[frame["uid"] == 1, "air_temperature_k"] = None
    frame.loc[frame["uid"] == 2, "type"] = None

    exact_duplicate = frame.loc[frame["uid"] == 3].copy()
    conflicting_record = frame.loc[frame["uid"] == 4].copy()
    conflicting_record.loc[:, "torque_nm"] += 1
    frame.loc[frame["uid"] == 5, "torque_nm"] = -1
    dirty = pd.concat([frame, exact_duplicate, conflicting_record], ignore_index=True)

    cleaned, quarantine, report = clean_dirty_demo_data(
        dirty,
        output_directory / "cleaned.csv",
        output_directory / "quarantine.csv",
        output_directory / "quality.json",
    )

    assert cleaned.shape == (3, 14)
    assert cleaned.loc[cleaned["uid"] == 1, "air_temperature_k"].iloc[0] == pytest.approx(300.5)
    assert cleaned.loc[cleaned["uid"] == 2, "type"].iloc[0] == "M"
    assert report["imputations"] == {"air_temperature_k": 1, "type": 1}
    assert report["actions"]["removed_exact_duplicate_rows"] == 1
    assert report["actions"]["quarantined_rows"] == 3
    assert sorted(quarantine["uid"].tolist()) == [4, 4, 5]
    assert "conflicting records" in " ".join(quarantine["quarantine_reason"])


@pytest.mark.parametrize("column", ["uid", "machine_failure"])
def test_dirty_demo_never_guesses_missing_identifiers_or_labels(column, output_directory):
    frame = pd.DataFrame([make_row(1), make_row(2)], columns=CANONICAL_COLUMNS)
    frame.loc[0, column] = None

    cleaned, quarantine, report = clean_dirty_demo_data(
        frame,
        output_directory / f"cleaned-{column}.csv",
        output_directory / f"quarantine-{column}.csv",
        output_directory / f"quality-{column}.json",
    )

    assert cleaned["uid"].tolist() == [2]
    assert len(quarantine) == 1
    assert report["actions"]["quarantined_rows"] == 1
    assert column in quarantine["quarantine_reason"].iloc[0]
