"""Tests for canonical AI4I schema and value validation."""

import numpy as np
import pandas as pd
import pytest

from industrial_analyzer.validation import CANONICAL_COLUMNS, validate_ai4i_data


def valid_frame():
    return pd.DataFrame(
        [
            {
                "uid": 1,
                "product_id": "M14860",
                "type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
                "machine_failure": 0,
                "twf": 0,
                "hdf": 0,
                "pwf": 0,
                "osf": 0,
                "rnf": 0,
            }
        ],
        columns=CANONICAL_COLUMNS,
    )


def test_accepts_valid_data_without_changing_it():
    frame = valid_frame()
    original = frame.copy(deep=True)

    assert validate_ai4i_data(frame) is None
    pd.testing.assert_frame_equal(frame, original)


def test_rejects_non_dataframe_input():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        validate_ai4i_data([1, 2, 3])


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="at least one row"):
        validate_ai4i_data(valid_frame().iloc[0:0])


def test_rejects_duplicate_column_names():
    frame = pd.concat([valid_frame(), valid_frame()[["uid"]]], axis=1)

    with pytest.raises(ValueError, match="Duplicate column names"):
        validate_ai4i_data(frame)


def test_rejects_missing_or_reordered_columns():
    missing = valid_frame().drop(columns="rnf")
    reordered = valid_frame()[list(reversed(CANONICAL_COLUMNS))]

    with pytest.raises(ValueError, match="exactly match"):
        validate_ai4i_data(missing)
    with pytest.raises(ValueError, match="exactly match"):
        validate_ai4i_data(reordered)


def test_rejects_missing_values():
    frame = valid_frame()
    frame.loc[0, "torque_nm"] = np.nan

    with pytest.raises(ValueError, match="Required values are missing"):
        validate_ai4i_data(frame)


def test_rejects_duplicate_identifiers():
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)

    with pytest.raises(ValueError, match="uid values must be unique"):
        validate_ai4i_data(frame)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("product_id", "", "must not contain empty strings"),
        ("type", "X", "type values must be exactly"),
        ("air_temperature_k", 0, "must be greater than zero"),
        ("process_temperature_k", -1, "must be greater than zero"),
        ("rotational_speed_rpm", 0, "must be greater than zero"),
        ("torque_nm", -0.1, "must be zero or greater"),
        ("tool_wear_min", -1, "must be zero or greater"),
        ("machine_failure", 2, "values must be 0 or 1"),
        ("twf", 2, "values must be 0 or 1"),
    ],
)
def test_rejects_invalid_values(column, value, message):
    frame = valid_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_ai4i_data(frame)


def test_rejects_infinite_measurements():
    frame = valid_frame()
    frame.loc[0, "torque_nm"] = np.inf

    with pytest.raises(ValueError, match="finite values"):
        validate_ai4i_data(frame)


def test_rejects_non_integer_binary_labels():
    frame = valid_frame()
    frame["machine_failure"] = frame["machine_failure"].astype(float)

    with pytest.raises(ValueError, match="integer labels 0 or 1"):
        validate_ai4i_data(frame)


def test_rejects_boolean_measurements_and_labels():
    measurement = valid_frame()
    measurement["torque_nm"] = True
    labels = valid_frame()
    labels["machine_failure"] = True

    with pytest.raises(ValueError, match="torque_nm must contain numeric values"):
        validate_ai4i_data(measurement)
    with pytest.raises(ValueError, match="machine_failure must contain integer labels"):
        validate_ai4i_data(labels)
