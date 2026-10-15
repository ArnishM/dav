"""Decision Tree training and single-record failure classification."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from industrial_analyzer.features import CANONICAL_COLUMNS, _validate_canonical


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_FEATURES = (
    "type", "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min",
)
METRICS_PATH = "outputs/reports/model_metrics.json"


def _resolve_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Output/model path must be a nonempty string")
    root = REPOSITORY_ROOT.resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Output/model path must stay inside the repository")
    return path


def _build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ("sensors", "passthrough", list(PREDICTION_FEATURES[1:])),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
    ])


def train_failure_model(df: pd.DataFrame, model_path: str) -> dict:
    """Validate, train and evaluate a tree; save its pipeline and JSON metadata."""
    _validate_canonical(df)
    destination = _resolve_path(model_path)
    report_path = _resolve_path(METRICS_PATH)
    if destination == report_path:
        raise ValueError("Model path must differ from the metrics JSON path")
    if df["machine_failure"].nunique() != 2:
        raise ValueError("machine_failure must contain both classes 0 and 1")

    # Separate the six raw inputs from all outcome labels and identifiers.
    inputs = df.loc[:, list(PREDICTION_FEATURES)].copy()
    target = df["machine_failure"].copy()
    try:
        train_inputs, test_inputs, train_target, test_target = train_test_split(
            inputs, target, test_size=0.2, stratify=target, random_state=42,
        )
    except ValueError as error:
        raise ValueError("Insufficient class observations for an 80/20 stratified split") from error
    if train_target.nunique() != 2 or test_target.nunique() != 2:
        raise ValueError("Both classes must appear in training and test partitions")
    folds = min(3, int(train_target.value_counts().min()))
    if folds < 2:
        raise ValueError("Insufficient training class observations for depth validation")

    # Choose among three depths using training-only stratified folds.
    search = GridSearchCV(
        _build_pipeline(), {"classifier__max_depth": [3, 5, 8]},
        scoring="f1", cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        error_score="raise",
    )
    try:
        search.fit(train_inputs, train_target)
        model = search.best_estimator_
        predictions = model.predict(test_inputs)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Sensor values cannot be processed by the Decision Tree: {error}") from error
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_target, predictions, average="binary", pos_label=1, zero_division=0,
    )
    metrics = {
        "model_name": "decision_tree", "evaluation_set": "held_out_test",
        "positive_class": 1, "random_state": 42, "test_size": 0.2,
        "train_samples": int(len(train_target)), "test_samples": int(len(test_target)),
        "failure_support": int((test_target == 1).sum()),
        "accuracy": float(accuracy_score(test_target, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix": confusion_matrix(test_target, predictions, labels=[0, 1]).tolist(),
    }
    result = {
        "schema_version": "1.0",
        "model_path": destination.relative_to(REPOSITORY_ROOT.resolve()).as_posix(),
        "metrics": metrics,
    }
    serialized_report = json.dumps(result, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the same fitted pipeline used for held-out evaluation.
    joblib.dump(model, destination)
    report_path.write_text(serialized_report + "\n", encoding="utf-8")
    return result


def _validate_readings(readings: dict) -> pd.DataFrame:
    if not isinstance(readings, dict) or set(readings) != set(PREDICTION_FEATURES):
        raise ValueError(f"readings must contain exactly these six keys: {PREDICTION_FEATURES}")
    # Reuse canonical domain checks with artificial ID/labels, never default sensors.
    row = {"uid": 1, "product_id": "validation-only", **readings}
    row.update({name: 0 for name in CANONICAL_COLUMNS[8:]})
    canonical = pd.DataFrame([row], columns=CANONICAL_COLUMNS, dtype=object)
    _validate_canonical(canonical)
    return canonical.loc[:, list(PREDICTION_FEATURES)]


def _load_pipeline(path: Path) -> Pipeline:
    try:
        model = joblib.load(path)
    except OSError:
        raise
    except Exception as error:
        raise ValueError("Invalid model artifact; expected a trusted fitted pipeline") from error
    try:
        if not isinstance(model, Pipeline) or list(model.named_steps) != ["preprocessing", "classifier"]:
            raise ValueError("Expected preprocessing/classifier Pipeline")
        preprocessing = model.named_steps["preprocessing"]
        classifier = model.named_steps["classifier"]
        if not isinstance(preprocessing, ColumnTransformer) or not isinstance(classifier, DecisionTreeClassifier):
            raise ValueError("Expected ColumnTransformer and DecisionTreeClassifier")
        check_is_fitted(preprocessing)
        check_is_fitted(classifier)
        if list(model.feature_names_in_) != list(PREDICTION_FEATURES):
            raise ValueError("Model input features do not match the six-feature contract")
        if set(classifier.classes_) != {0, 1}:
            raise ValueError("Model must classify both binary classes")
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"Incompatible model artifact: {error}") from error
    return model


def predict_one(readings: dict, model_path: str) -> dict:
    """Classify six raw inputs using a trusted saved pipeline, without retraining."""
    inputs = _validate_readings(readings)
    model = _load_pipeline(_resolve_path(model_path))
    try:
        predictions = model.predict(inputs)
        if len(predictions) != 1 or predictions[0] not in (0, 1):
            raise ValueError("Model returned an invalid binary prediction")
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(inputs), dtype=float)
            classes = model.named_steps["classifier"].classes_
            if probabilities.shape != (1, len(classes)) or not np.isfinite(probabilities).all():
                raise ValueError("Model returned invalid probabilities")
            if (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1):
                raise ValueError("Model probabilities must be in [0, 1] and sum to one")
            # Find the failure column by its label, not by an assumed position.
            failure_column = list(classes).index(1)
            probability = float(probabilities[0, failure_column])
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError(f"Cannot classify readings: {error}") from error
    return {
        "schema_version": "1.0", "predicted_failure": int(predictions[0]),
        "failure_probability": probability, "model_name": "decision_tree",
        "message": "Model classification; not a certified equipment diagnosis",
    }
