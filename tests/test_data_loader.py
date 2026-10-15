"""Tests for loading the official AI4I CSV into the canonical schema."""

import pandas as pd
import pytest

from industrial_analyzer.data_loader import load_ai4i_csv
from industrial_analyzer.validation import CANONICAL_COLUMNS


SOURCE_ROW = {
    "UDI": 1,
    "Product ID": "M14860",
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
    "Machine failure": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0,
}


def test_loads_official_headers_and_maps_to_canonical_columns(tmp_path):
    path = tmp_path / "ai4i.csv"
    pd.DataFrame([SOURCE_ROW]).to_csv(path, index=False)
    original_bytes = path.read_bytes()

    loaded = load_ai4i_csv(path)

    assert loaded.columns.tolist() == CANONICAL_COLUMNS
    assert loaded.loc[0, "uid"] == 1
    assert loaded.loc[0, "product_id"] == "M14860"
    assert loaded.loc[0, "air_temperature_k"] == pytest.approx(298.1)
    assert path.read_bytes() == original_bytes


def test_loads_a_canonical_csv_without_renaming(tmp_path):
    row = {
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
    path = tmp_path / "canonical.csv"
    pd.DataFrame([row], columns=CANONICAL_COLUMNS).to_csv(path, index=False)

    loaded = load_ai4i_csv(path)

    assert loaded.columns.tolist() == CANONICAL_COLUMNS
    assert len(loaded) == 1


def test_rejects_duplicate_csv_headers(tmp_path):
    path = tmp_path / "duplicate_headers.csv"
    path.write_text("uid,uid\n1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_ai4i_csv(path)


def test_rejects_unrecognized_headers(tmp_path):
    path = tmp_path / "wrong_headers.csv"
    path.write_text("first,second\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="headers must match"):
        load_ai4i_csv(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ai4i_csv(tmp_path / "missing.csv")


@pytest.mark.parametrize("path", ["", "   "])
def test_rejects_empty_path(path):
    with pytest.raises(ValueError, match="nonempty"):
        load_ai4i_csv(path)
