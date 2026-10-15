"""Artificial rows only: these are not records from the UCI dataset."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from industrial_analyzer.features import CANONICAL_COLUMNS, engineer_features


@pytest.fixture
def artificial_data():
    return pd.DataFrame(
        [
            [2, "ARTIFICIAL-M", "M", 300.0, 312.0, 1500, 40, 0, 0, 0, 0, 0, 0, 0],
            [1, "ARTIFICIAL-L", "L", 295.0, 304.5, 1200, 50, 100, 1, 1, 0, 0, 0, 0],
        ],
        columns=CANONICAL_COLUMNS,
        index=pd.Index(["second", "first"], name="artificial_row"),
    )


def test_calculations_and_preservation(artificial_data):
    original = artificial_data.copy(deep=True)
    result = engineer_features(artificial_data)
    assert result is not artificial_data
    assert list(result.columns) == list(CANONICAL_COLUMNS) + [
        "temperature_difference_k", "torque_speed_product"
    ]
    assert_frame_equal(result.iloc[:, :14], original)
    assert_frame_equal(artificial_data, original)
    assert result["temperature_difference_k"].tolist() == [12.0, 9.5]
    assert result["torque_speed_product"].tolist() == [60000.0, 60000.0]


@pytest.mark.parametrize("column", CANONICAL_COLUMNS)
def test_missing_column(artificial_data, column):
    with pytest.raises(ValueError, match="canonical columns"):
        engineer_features(artificial_data.drop(columns=column))


@pytest.mark.parametrize("column,value,message", [
    ("type", "X", "type"),
    ("uid", 2, "unique"),
    ("uid", 1.5, "integers"),
    ("product_id", "", "product_id"),
    ("machine_failure", 2, "binary"),
    ("twf", True, "integers"),
    ("torque_nm", "40", "real numbers"),
    ("tool_wear_min", False, "real numbers"),
])
def test_invalid_values(artificial_data, column, value, message):
    data = artificial_data.astype({column: object})
    data.loc["first", column] = value
    with pytest.raises(ValueError, match=message):
        engineer_features(data)


@pytest.mark.parametrize("column", CANONICAL_COLUMNS[3:8])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_invalid_sensors(artificial_data, column, value):
    data = artificial_data.astype({column: float})
    data.loc["first", column] = value
    with pytest.raises(ValueError, match=column):
        engineer_features(data)


@pytest.mark.parametrize("column", CANONICAL_COLUMNS[3:6])
def test_zero_invalid_positive_sensors(artificial_data, column):
    artificial_data.loc["first", column] = 0
    with pytest.raises(ValueError, match=column):
        engineer_features(artificial_data)


@pytest.mark.parametrize("column", ["temperature_difference_k", "torque_speed_product"])
def test_derived_collision(artificial_data, column):
    artificial_data[column] = 0
    with pytest.raises(ValueError, match="collision"):
        engineer_features(artificial_data)


def test_empty_input(artificial_data):
    with pytest.raises(ValueError, match="at least one row"):
        engineer_features(artificial_data.iloc[:0])


def test_wrong_input_type():
    with pytest.raises(ValueError, match="DataFrame"):
        engineer_features(None)


def test_column_order_and_extra_column(artificial_data):
    with pytest.raises(ValueError, match="contract order"):
        engineer_features(artificial_data[list(reversed(CANONICAL_COLUMNS))])
    with pytest.raises(ValueError, match="unexpected"):
        engineer_features(artificial_data.assign(unrelated=1))


def test_duplicate_columns(artificial_data):
    data = pd.concat([artificial_data, artificial_data[["uid"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        engineer_features(data)


def test_calculated_overflow(artificial_data):
    data = artificial_data.astype({"torque_nm": float})
    data.loc["first", "torque_nm"] = 1e308
    with pytest.raises(ValueError, match="Calculated torque_speed_product"):
        engineer_features(data)


def test_integer_product_does_not_wrap(artificial_data):
    artificial_data.loc["first", "torque_nm"] = 2**62
    result = engineer_features(artificial_data)
    assert result.loc["first", "torque_speed_product"] == float(2**62) * 1200


def test_preserves_extremes_negative_difference_and_duplicate_index(artificial_data):
    artificial_data.index = [7, 7]
    artificial_data.loc[:, "process_temperature_k"] = 290.0
    artificial_data.loc[:, "torque_nm"] = 1000000
    result = engineer_features(artificial_data)
    assert_frame_equal(result.iloc[:, :14], artificial_data)
    assert result["temperature_difference_k"].tolist() == [-10.0, -5.0]
