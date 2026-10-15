"""Tests for separate transformed data and descriptive tables."""

import pandas as pd
import pytest

from industrial_analyzer.statistics import (
    failure_by_type_table,
    failure_modes_table,
    summary_statistics_table,
    write_statistics_tables,
)
from industrial_analyzer.transformation import transform_for_dav, write_transformed_csv
from industrial_analyzer.validation import CANONICAL_COLUMNS


def make_frame() -> pd.DataFrame:
    rows = []
    for uid, category, failure in [(1, "L", 0), (2, "M", 1), (3, "L", 1)]:
        rows.append(
            [uid, f"P{uid}", category, 298.0 + uid, 308.0 + uid,
             1400 + uid, 30.0 + uid, 10 * uid, failure, failure, 0, 0, 0, 0]
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def test_transformation_preserves_canonical_data_and_separates_demo_columns():
    frame = make_frame()
    original = frame.copy(deep=True)

    transformed = transform_for_dav(frame)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(transformed[CANONICAL_COLUMNS], frame)
    assert transformed[["type_L", "type_M", "type_H"]].values.tolist() == [
        [1, 0, 0], [0, 1, 0], [1, 0, 0]
    ]
    for column in [f"{name}_minmax" for name in [
        "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min",
    ]]:
        assert transformed[column].min() == pytest.approx(0)
        assert transformed[column].max() == pytest.approx(1)
    assert transformed["tool_wear_quantile_bin"].min() >= 1


def test_statistics_tables_have_explicit_columns_and_correct_counts():
    frame = make_frame()

    summary = summary_statistics_table(frame)
    by_type = failure_by_type_table(frame)
    modes = failure_modes_table(frame)

    assert summary.columns.tolist() == [
        "feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]
    assert by_type.columns.tolist() == [
        "type", "total_records", "failure_count", "nonfailure_count", "failure_rate_percent"
    ]
    assert by_type.loc[by_type["type"] == "L", "failure_count"].iloc[0] == 1
    assert by_type.loc[by_type["type"] == "M", "failure_rate_percent"].iloc[0] == 100
    assert by_type.loc[by_type["type"] == "H", "total_records"].iloc[0] == 0
    assert modes["failure_mode"].tolist() == ["twf", "hdf", "pwf", "osf", "rnf"]
    assert modes.loc[modes["failure_mode"] == "twf", "flagged_records"].iloc[0] == 2


def test_transform_handles_constant_sensor_values():
    frame = make_frame()
    frame["tool_wear_min"] = 10
    frame["torque_nm"] = 5

    transformed = transform_for_dav(frame)

    assert transformed["tool_wear_quantile_bin"].tolist() == [1, 1, 1]
    assert transformed["tool_wear_min_minmax"].tolist() == [0.0, 0.0, 0.0]
    assert transformed["tool_wear_min_zscore"].tolist() == [0.0, 0.0, 0.0]


def test_writers_use_agreed_paths_and_no_index_column(tmp_path):
    frame = make_frame()
    transformed_path = tmp_path / "ai4i_transformed.csv"
    transformed = write_transformed_csv(frame, str(transformed_path))
    output_dir = tmp_path / "tables"
    paths = write_statistics_tables(frame, output_dir)

    assert pd.read_csv(transformed_path).columns.tolist() == transformed.columns.tolist()
    assert set(paths) == {"summary_statistics", "failure_by_type", "failure_modes"}
    for path in paths.values():
        assert path.exists()
        assert not any(column.startswith("Unnamed:") for column in pd.read_csv(path).columns)
