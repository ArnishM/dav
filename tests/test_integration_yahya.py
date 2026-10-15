"""Yahya's module connections; artificial fixtures are not project results."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from industrial_analyzer import advanced_analysis, prediction
from industrial_analyzer.advanced_analysis import plot_failure_insights
from industrial_analyzer.features import CANONICAL_COLUMNS, engineer_features
from industrial_analyzer.prediction import predict_one, train_failure_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_INPUTS = [
    "type", "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min",
]
EXPECTED_FIGURES = [
    "fig_y_01_torque_by_failure.png",
    "fig_y_02_torque_vs_rpm.png",
    "fig_y_03_failure_rate_by_type.png",
]


@pytest.fixture
def isolated_repository(tmp_path, monkeypatch):
    # Redirect BOTH roots, including the fixed training metrics destination.
    monkeypatch.setattr(prediction, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(advanced_analysis, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def artificial_clean_data():
    rows = []
    for number in range(60):
        failure = int(number % 5 == 0)
        rows.append([
            number + 1, f"ARTIFICIAL-{number}", ("L", "M", "H")[number % 3],
            298.0 + number / 100, 308.0 + number / 100, 1400 + number,
            65.0 if failure else 35.0, number, failure, failure, 0, 0, 0, 0,
        ])
    return pd.DataFrame(
        rows, columns=CANONICAL_COLUMNS,
        index=pd.Index(range(100, 160), name="artificial_record"),
    )


def _run_workflow(clean_df, readings):
    features = engineer_features(clean_df)
    # Train on the original 14 columns; derived columns are descriptive only.
    model_info = train_failure_model(clean_df, "models/failure_model.joblib")
    result = predict_one(readings, model_info["model_path"])
    figure_paths = plot_failure_insights(features, "outputs/figures")
    return features, model_info, result, figure_paths


def _check_outputs(root, clean_df, outputs):
    features, model_info, result, figure_paths = outputs
    assert list(features.columns) == list(CANONICAL_COLUMNS) + [
        "temperature_difference_k", "torque_speed_product",
    ]
    assert features is not clean_df
    assert_frame_equal(features.iloc[:, :14], clean_df)
    assert set(model_info) == {"schema_version", "model_path", "metrics"}
    assert model_info["schema_version"] == "1.0"
    assert model_info["model_path"] == "models/failure_model.joblib"
    saved_report = json.loads((root / "outputs/reports/model_metrics.json").read_text(encoding="utf-8"))
    assert saved_report == model_info

    model = joblib.load(root / model_info["model_path"])
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["preprocessing"], ColumnTransformer)
    assert isinstance(model.named_steps["classifier"], DecisionTreeClassifier)
    assert list(model.feature_names_in_) == APPROVED_INPUTS
    assert list(model.named_steps["preprocessing"].feature_names_in_) == APPROVED_INPUTS
    transformer_columns = [
        column for name, _, columns in model.named_steps["preprocessing"].transformers_
        if name != "remainder" for column in columns
    ]
    assert transformer_columns == APPROVED_INPUTS

    metrics = model_info["metrics"]
    assert metrics["random_state"] == 42 and metrics["test_size"] == 0.2
    assert metrics["positive_class"] == 1
    assert metrics["train_samples"] + metrics["test_samples"] == len(clean_df)
    assert metrics["test_samples"] == int(np.ceil(len(clean_df) * 0.2))
    matrix = np.asarray(metrics["confusion_matrix"])
    assert matrix.shape == (2, 2) and (matrix >= 0).all()
    assert matrix.sum() == metrics["test_samples"]
    assert matrix[1].sum() == metrics["failure_support"]
    assert metrics["accuracy"] == pytest.approx(np.trace(matrix) / matrix.sum())

    assert set(result) == {"schema_version", "predicted_failure", "failure_probability", "model_name", "message"}
    assert result["schema_version"] == "1.0"
    assert type(result["predicted_failure"]) is int and result["predicted_failure"] in (0, 1)
    probability = result["failure_probability"]
    assert probability is None or (type(probability) is float and np.isfinite(probability) and 0 <= probability <= 1)
    assert type(result["model_name"]) is str and result["model_name"] == "decision_tree"
    assert type(result["message"]) is str
    json.dumps(result, allow_nan=False)
    json.dumps(model_info, allow_nan=False)

    assert figure_paths == [f"outputs/figures/{name}" for name in EXPECTED_FIGURES]
    for relative_path in figure_paths:
        assert type(relative_path) is str and "\\" not in relative_path
        image_path = root / relative_path
        assert image_path.stat().st_size > 0
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            image.verify()
        with Image.open(image_path) as image:
            image.load()
            assert image.width > 100 and image.height > 100
    return model


@pytest.mark.parametrize("nested_working_directory", [False, True])
def test_end_to_end_repeatable_and_preserves_files(
    artificial_clean_data, isolated_repository, monkeypatch, nested_working_directory,
):
    root = isolated_repository
    if nested_working_directory:
        notebook_directory = root / "notebooks/demo"
        notebook_directory.mkdir(parents=True)
        monkeypatch.chdir(notebook_directory)
    sentinels = {
        "outputs/figures/fig_01_amirul.png": b"unrelated chart",
        "outputs/reports/data_quality.json": b'{"untouched": true}',
        "data/processed/ai4i_cleaned.csv": b"unrelated dataset sentinel",
    }
    for relative_path, content in sentinels.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    clean_df = artificial_clean_data
    original = clean_df.copy(deep=True)
    readings = clean_df.loc[clean_df.index[0], APPROVED_INPUTS].to_dict()
    first = _run_workflow(clean_df, readings)
    first_model = _check_outputs(root, clean_df, first)
    first_predictions = first_model.predict(clean_df[APPROVED_INPUTS])
    first_png_bytes = [(root / path).read_bytes() for path in first[3]]
    second = _run_workflow(clean_df, readings)
    second_model = _check_outputs(root, clean_df, second)
    assert_frame_equal(first[0], second[0])
    assert first[1:] == second[1:]
    np.testing.assert_array_equal(first_predictions, second_model.predict(clean_df[APPROVED_INPUTS]))
    assert first_png_bytes == [(root / path).read_bytes() for path in second[3]]
    assert len(list((root / "outputs/figures").glob("fig_y_*.png"))) == 3
    assert_frame_equal(clean_df, original)
    for relative_path, content in sentinels.items():
        assert (root / relative_path).read_bytes() == content


def test_workflow_failure_propagates_without_artifacts(artificial_clean_data, isolated_repository):
    invalid = artificial_clean_data.copy()
    invalid["machine_failure"] = 0
    readings = invalid.loc[invalid.index[0], APPROVED_INPUTS].to_dict()
    with pytest.raises(ValueError, match="both classes"):
        _run_workflow(invalid, readings)
    assert not (isolated_repository / "models/failure_model.joblib").exists()
    assert not (isolated_repository / "outputs/reports/model_metrics.json").exists()
    assert not (isolated_repository / "outputs/figures").exists()


def test_engineered_data_is_not_training_input(artificial_clean_data, isolated_repository):
    features = engineer_features(artificial_clean_data)
    with pytest.raises(ValueError, match="collision"):
        train_failure_model(features, "models/failure_model.joblib")
    assert not (isolated_repository / "models/failure_model.joblib").exists()


def test_amirul_cleaned_csv_read_only(isolated_repository):
    path = PROJECT_ROOT / "data/processed/ai4i_cleaned.csv"
    if not path.is_file():
        pytest.skip("Amirul's canonical cleaned CSV is unavailable; compatibility test pending")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        clean_df = pd.read_csv(path, encoding="utf-8")
        assert list(clean_df.columns) == list(CANONICAL_COLUMNS)
        for column in ("uid", *CANONICAL_COLUMNS[8:]):
            assert pd.api.types.is_integer_dtype(clean_df[column]), column
        for column in CANONICAL_COLUMNS[3:8]:
            assert pd.api.types.is_numeric_dtype(clean_df[column]), column
        original = clean_df.copy(deep=True)
        readings = clean_df.loc[clean_df.index[0], APPROVED_INPUTS].to_dict()
        outputs = _run_workflow(clean_df, readings)
        _check_outputs(isolated_repository, clean_df, outputs)
        assert_frame_equal(clean_df, original)
    finally:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
