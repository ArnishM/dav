"""Prediction contract tests using artificial rows and isolated artifacts."""

import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from industrial_analyzer import prediction
from industrial_analyzer.features import CANONICAL_COLUMNS


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "REPOSITORY_ROOT", tmp_path)


@pytest.fixture
def artificial_data():
    rows = []
    for i in range(100):
        failure = int(i % 5 == 0)
        rows.append([
            i + 1, f"ARTIFICIAL-{i}", ("L", "M", "H")[i % 3],
            298.0 + i / 100, 310.0 + i / 100, 1500 + i,
            70.0 if failure else 30.0, i, failure, failure, 0, 0, 0, 0,
        ])
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


@pytest.fixture
def readings():
    return dict(zip(prediction.PREDICTION_FEATURES, ["L", 298.1, 308.6, 1551, 42.8, 0]))


@pytest.fixture
def trained(artificial_data):
    return prediction.train_failure_model(artificial_data, "models/failure_model.joblib")


def test_signatures():
    assert str(inspect.signature(prediction.train_failure_model)) == "(df: pandas.DataFrame, model_path: str) -> dict"
    assert str(inspect.signature(prediction.predict_one)) == "(readings: dict, model_path: str) -> dict"


def test_training_metadata_serialization_and_evaluation(artificial_data, trained):
    root = prediction.REPOSITORY_ROOT
    assert set(trained) == {"schema_version", "model_path", "metrics"}
    assert trained["schema_version"] == "1.0"
    assert json.loads((root / prediction.METRICS_PATH).read_text()) == trained
    metrics = trained["metrics"]
    assert set(metrics) == {
        "model_name", "evaluation_set", "positive_class", "random_state", "test_size",
        "train_samples", "test_samples", "failure_support", "accuracy", "precision", "recall", "f1", "confusion_matrix",
    }
    assert metrics["train_samples"] == 80 and metrics["test_samples"] == 20
    assert metrics["failure_support"] == 4 and metrics["positive_class"] == 1
    model = joblib.load(root / trained["model_path"])
    assert list(model.feature_names_in_) == list(prediction.PREDICTION_FEATURES)
    assert list(model.named_steps["preprocessing"].feature_names_in_) == list(prediction.PREDICTION_FEATURES)
    assert model.named_steps["classifier"].max_depth in (3, 5, 8)
    _, test_inputs, _, test_target = train_test_split(
        artificial_data[list(prediction.PREDICTION_FEATURES)], artificial_data.machine_failure,
        test_size=0.2, stratify=artificial_data.machine_failure, random_state=42,
    )
    expected = confusion_matrix(test_target, model.predict(test_inputs), labels=[0, 1]).tolist()
    assert metrics["confusion_matrix"] == expected
    assert sum(map(sum, expected)) == 20 and sum(expected[1]) == 4
    for name in ("accuracy", "precision", "recall", "f1"):
        assert type(metrics[name]) is float and 0 <= metrics[name] <= 1
    json.dumps(trained, allow_nan=False)


def test_no_mutation_and_reproducibility(artificial_data):
    original = artificial_data.copy(deep=True)
    first = prediction.train_failure_model(artificial_data, "models/first.joblib")
    second = prediction.train_failure_model(artificial_data, "models/second.joblib")
    assert first["metrics"] == second["metrics"]
    assert_frame_equal(artificial_data, original)
    a = joblib.load(prediction.REPOSITORY_ROOT / first["model_path"])
    b = joblib.load(prediction.REPOSITORY_ROOT / second["model_path"])
    np.testing.assert_array_equal(a.predict(artificial_data[list(prediction.PREDICTION_FEATURES)]), b.predict(artificial_data[list(prediction.PREDICTION_FEATURES)]))


def test_no_identifier_or_flag_leakage(artificial_data, trained):
    modified = artificial_data.copy()
    modified["uid"] = np.arange(1000, 1100)
    modified["product_id"] = "OTHER-ARTIFICIAL"
    modified.loc[:, ["twf", "hdf", "pwf", "osf", "rnf"]] = 1
    other = prediction.train_failure_model(modified, "models/other.joblib")
    assert other["metrics"] == trained["metrics"]


def test_prediction_roundtrip(readings, trained, monkeypatch):
    # Prediction must not fit or retrain anything.
    monkeypatch.setattr(prediction.GridSearchCV, "fit", lambda *a, **k: pytest.fail("Unexpected fit"))
    original = readings.copy()
    result = prediction.predict_one(readings, trained["model_path"])
    assert readings == original
    assert set(result) == {"schema_version", "predicted_failure", "failure_probability", "model_name", "message"}
    assert type(result["predicted_failure"]) is int and result["predicted_failure"] in (0, 1)
    assert type(result["failure_probability"]) is float and 0 <= result["failure_probability"] <= 1
    assert result["schema_version"] == "1.0" and result["model_name"] == "decision_tree"
    assert result["message"] == "Model classification; not a certified equipment diagnosis"
    assert prediction.predict_one(dict(reversed(list(readings.items()))), trained["model_path"]) == result
    json.dumps(result, allow_nan=False)


def test_positive_probability_uses_labels(readings, trained, monkeypatch):
    model = joblib.load(prediction.REPOSITORY_ROOT / trained["model_path"])
    model.named_steps["classifier"].classes_ = np.array([1, 0])
    monkeypatch.setattr(model, "predict", lambda x: np.array([1]))
    monkeypatch.setattr(model, "predict_proba", lambda x: np.array([[0.8, 0.2]]))
    monkeypatch.setattr(prediction, "_load_pipeline", lambda path: model)
    assert prediction.predict_one(readings, trained["model_path"])["failure_probability"] == 0.8


@pytest.mark.parametrize("column,value", [
    ("type", "X"), ("air_temperature_k", 0), ("process_temperature_k", -1),
    ("rotational_speed_rpm", 0), ("torque_nm", -1), ("tool_wear_min", -1),
    ("torque_nm", "40"), ("torque_nm", True), ("torque_nm", np.nan),
    ("torque_nm", np.inf), ("torque_nm", None),
])
def test_invalid_readings(readings, column, value):
    readings[column] = value
    with pytest.raises(ValueError, match=column):
        prediction.predict_one(readings, "models/missing.joblib")


@pytest.mark.parametrize("key", prediction.PREDICTION_FEATURES)
def test_missing_prediction_keys(readings, key):
    del readings[key]
    with pytest.raises(ValueError, match="six keys"):
        prediction.predict_one(readings, "models/missing.joblib")


def test_extra_keys_and_wrong_type(readings):
    with pytest.raises(ValueError, match="six keys"):
        prediction.predict_one({**readings, "twf": 1}, "models/missing.joblib")
    with pytest.raises(ValueError, match="six keys"):
        prediction.predict_one(None, "models/missing.joblib")


def test_missing_file(readings):
    with pytest.raises(FileNotFoundError):
        prediction.predict_one(readings, "models/missing.joblib")


@pytest.mark.parametrize("artifact", [None, {"wrong": "artifact"}, prediction._build_pipeline()])
def test_incompatible_artifact(readings, artifact):
    path = prediction.REPOSITORY_ROOT / "bad.joblib"
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="Incompatible"):
        prediction.predict_one(readings, "bad.joblib")


def test_corrupt_artifact(readings):
    (prediction.REPOSITORY_ROOT / "bad.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="Invalid model artifact"):
        prediction.predict_one(readings, "bad.joblib")


@pytest.mark.parametrize("size", [0, 1, 2, 5])
def test_insufficient_training(artificial_data, size):
    with pytest.raises(ValueError):
        prediction.train_failure_model(artificial_data.iloc[:size], "model.joblib")


def test_single_class(artificial_data):
    artificial_data["machine_failure"] = 0
    with pytest.raises(ValueError, match="both classes"):
        prediction.train_failure_model(artificial_data, "model.joblib")


@pytest.mark.parametrize("column,value", [("type", "X"), ("torque_nm", -1), ("machine_failure", 2)])
def test_invalid_training(artificial_data, column, value):
    artificial_data.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        prediction.train_failure_model(artificial_data, "model.joblib")


def test_schema_validation(artificial_data):
    with pytest.raises(ValueError, match="canonical columns"):
        prediction.train_failure_model(artificial_data.drop(columns="uid"), "model.joblib")
    with pytest.raises(ValueError, match="collision"):
        prediction.train_failure_model(artificial_data.assign(temperature_difference_k=1), "model.joblib")


@pytest.mark.parametrize("path", ["", "../escape.joblib", "outputs/reports/model_metrics.json"])
def test_invalid_model_paths(artificial_data, path):
    with pytest.raises(ValueError):
        prediction.train_failure_model(artificial_data, path)


def test_absolute_paths_and_nested_working_directory(artificial_data, readings, monkeypatch):
    root = prediction.REPOSITORY_ROOT
    nested = root / "notebooks"
    nested.mkdir()
    monkeypatch.chdir(nested)
    result = prediction.train_failure_model(artificial_data, str(root / "models/nested/model.joblib"))
    assert result["model_path"] == "models/nested/model.joblib"
    assert (root / prediction.METRICS_PATH).is_file()
    prediction.predict_one(readings, result["model_path"])


def test_training_only_preprocessing(artificial_data, monkeypatch):
    from sklearn.preprocessing import OneHotEncoder
    _, test, _, _ = train_test_split(
        artificial_data, artificial_data.machine_failure,
        test_size=0.2, stratify=artificial_data.machine_failure, random_state=42,
    )
    artificial_data.loc[test.index, "type"] = "H"
    artificial_data.loc[artificial_data.index.difference(test.index), "type"] = "L"
    sizes = []
    original_fit = OneHotEncoder.fit
    def record_fit(self, x, y=None):
        sizes.append(len(x))
        return original_fit(self, x, y)
    monkeypatch.setattr(OneHotEncoder, "fit", record_fit)
    result = prediction.train_failure_model(artificial_data, "model.joblib")
    model = joblib.load(prediction.REPOSITORY_ROOT / result["model_path"])
    assert model.named_steps["preprocessing"].named_transformers_["type"].categories_[0].tolist() == ["L"]
    assert max(sizes) == 80 and all(size < 100 for size in sizes)


@pytest.mark.parametrize("problem", ["features", "classes"])
def test_incompatible_fitted_model(readings, trained, problem):
    path = prediction.REPOSITORY_ROOT / trained["model_path"]
    model = joblib.load(path)
    if problem == "features":
        model.named_steps["preprocessing"].feature_names_in_[0] = "uid"
    else:
        model.named_steps["classifier"].classes_ = np.array([0, 2])
    joblib.dump(model, path)
    with pytest.raises(ValueError, match="Incompatible"):
        prediction.predict_one(readings, trained["model_path"])


@pytest.mark.parametrize("probabilities", [[[np.nan, 0.5]], [[-0.1, 1.1]], [[0.2, 0.2]], [[1.0]]])
def test_invalid_model_probabilities(readings, trained, monkeypatch, probabilities):
    model = joblib.load(prediction.REPOSITORY_ROOT / trained["model_path"])
    monkeypatch.setattr(model, "predict_proba", lambda x: np.asarray(probabilities))
    monkeypatch.setattr(prediction, "_load_pipeline", lambda path: model)
    with pytest.raises(ValueError, match="probabilit"):
        prediction.predict_one(readings, trained["model_path"])


def test_invalid_model_prediction(readings, trained, monkeypatch):
    model = joblib.load(prediction.REPOSITORY_ROOT / trained["model_path"])
    monkeypatch.setattr(model, "predict", lambda x: np.array([2]))
    monkeypatch.setattr(prediction, "_load_pipeline", lambda path: model)
    with pytest.raises(ValueError, match="invalid binary"):
        prediction.predict_one(readings, trained["model_path"])
