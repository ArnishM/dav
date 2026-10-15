"""Run Amirul's preparation and analytics, then connect Yahya's functions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

from industrial_analyzer.cleaning import clean_official_ai4i_data
from industrial_analyzer.data_loader import load_ai4i_csv
from industrial_analyzer.statistics import write_statistics_tables
from industrial_analyzer.transformation import write_transformed_csv
from industrial_analyzer.visualization import plot_main_insights


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "outputs/reports/run_manifest.json"


def _write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _load_yahya_modules():
    try:
        features = importlib.import_module("industrial_analyzer.features")
        prediction = importlib.import_module("industrial_analyzer.prediction")
        advanced_analysis = importlib.import_module("industrial_analyzer.advanced_analysis")
    except ModuleNotFoundError as exc:
        if exc.name in {
            "industrial_analyzer.features",
            "industrial_analyzer.prediction",
            "industrial_analyzer.advanced_analysis",
        }:
            return None
        raise
    return features, prediction, advanced_analysis


def run_pipeline(readings: dict | None = None) -> dict[str, Any]:
    """Run the full pipeline, or report the missing Yahya integration clearly.

    ``readings`` is an optional six-field dictionary for ``predict_one``. The
    caller must provide it; this function never invents sensor readings.
    """
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "inputs": {"source_csv": "data/raw/ai4i2020.csv"},
        "outputs": {},
        "completed_steps": [],
        "pending_steps": [],
        "errors": [],
    }
    result: dict[str, Any] = {"manifest_path": "outputs/reports/run_manifest.json"}
    current_step = "official_cleaning"

    try:
        cleaned = clean_official_ai4i_data()
        manifest["completed_steps"].append("official_cleaning_and_quality_report")
        manifest["outputs"]["cleaned_csv"] = "data/processed/ai4i_cleaned.csv"
        manifest["outputs"]["data_quality_report"] = "outputs/reports/data_quality.json"

        current_step = "canonical_load_and_validation"
        clean_path = REPOSITORY_ROOT / "data/processed/ai4i_cleaned.csv"
        clean = load_ai4i_csv(clean_path)
        if len(clean) != len(cleaned):
            raise ValueError("Reloaded canonical CSV row count differs from cleaned data")
        manifest["completed_steps"].append("canonical_load_and_validation")

        current_step = "transformation"
        transformed = write_transformed_csv(clean)
        manifest["outputs"]["transformed_csv"] = "data/processed/ai4i_transformed.csv"
        manifest["completed_steps"].append("descriptive_transformation")

        current_step = "statistics"
        table_paths = write_statistics_tables(clean)
        manifest["outputs"]["tables"] = {
            name: _relative(path) for name, path in table_paths.items()
        }
        manifest["completed_steps"].append("descriptive_statistics")

        current_step = "amirul_charts"
        figure_paths = plot_main_insights(clean)
        manifest["outputs"]["amirul_figures"] = figure_paths
        manifest["completed_steps"].append("amirul_descriptive_figures")

        result.update({
            "cleaned_rows": len(clean),
            "transformed_columns": len(transformed.columns),
            "tables": manifest["outputs"]["tables"],
            "amirul_figures": figure_paths,
        })

        current_step = "yahya_integration"
        modules = _load_yahya_modules()
        if modules is None:
            manifest["status"] = "awaiting_yahya"
            manifest["pending_steps"] = [
                "engineer_features",
                "train_failure_model",
                "predict_one",
                "plot_failure_insights",
            ]
            manifest["errors"].append({
                "step": current_step,
                "message": "Yahya's modules are not present in the current amirul checkout.",
            })
        else:
            features_module, prediction_module, charts_module = modules
            engineered = features_module.engineer_features(clean)
            manifest["completed_steps"].append("yahya_feature_engineering")

            model_info = prediction_module.train_failure_model(
                clean, "models/failure_model.joblib"
            )
            manifest["outputs"]["model"] = model_info["model_path"]
            manifest["outputs"]["model_metrics"] = "outputs/reports/model_metrics.json"
            manifest["completed_steps"].append("yahya_model_training_and_evaluation")

            advanced_paths = charts_module.plot_failure_insights(
                engineered, "outputs/figures"
            )
            manifest["outputs"]["yahya_figures"] = advanced_paths
            manifest["completed_steps"].append("yahya_failure_figures")
            result["model_info"] = model_info

            if readings is None:
                manifest["status"] = "awaiting_readings"
                manifest["pending_steps"] = ["predict_one; supply six sensor readings"]
            else:
                prediction = prediction_module.predict_one(
                    readings, model_info["model_path"]
                )
                result["prediction"] = prediction
                manifest["outputs"]["single_reading_prediction"] = "returned_to_caller"
                manifest["completed_steps"].append("yahya_single_reading_prediction")
                manifest["status"] = "complete"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append({
            "step": current_step,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        _write_manifest(manifest)
        raise

    _write_manifest(manifest)
    result["status"] = manifest["status"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings-file",
        help="Optional UTF-8 JSON object containing exactly six approved sensor fields",
    )
    args = parser.parse_args()
    readings = None
    if args.readings_file:
        path = Path(args.readings_file).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        with path.open("r", encoding="utf-8") as handle:
            readings = json.load(handle)
        if not isinstance(readings, dict):
            raise ValueError("The readings JSON file must contain one JSON object")

    print(json.dumps(run_pipeline(readings), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
