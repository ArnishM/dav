"""Tests for pipeline manifest behavior before Yahya's modules are integrated."""

import json

import pandas as pd

from industrial_analyzer.validation import CANONICAL_COLUMNS
from scripts import run_pipeline as pipeline


def make_frame() -> pd.DataFrame:
    rows = []
    for uid in range(1, 5):
        failure = uid % 2
        rows.append([
            uid, f"P{uid}", ("L", "M", "H", "L")[uid - 1],
            298.0 + uid, 308.0 + uid, 1400 + uid, 30.0 + uid,
            uid * 10, failure, failure, 0, 0, 0, 0,
        ])
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def test_pipeline_records_own_outputs_and_waits_for_yahya_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", tmp_path / "outputs/reports/run_manifest.json")
    clean = make_frame()

    def write_cleaned():
        return clean.copy()

    monkeypatch.setattr(pipeline, "clean_official_ai4i_data", write_cleaned)
    monkeypatch.setattr(pipeline, "load_ai4i_csv", lambda _path: clean.copy())
    monkeypatch.setattr(pipeline, "write_transformed_csv", lambda _df: pd.DataFrame(columns=range(28)))
    monkeypatch.setattr(
        pipeline, "write_statistics_tables",
        lambda _df: {name: tmp_path / "outputs/tables" / f"{name}.csv"
                     for name in ["summary_statistics", "failure_by_type", "failure_modes"]},
    )
    monkeypatch.setattr(
        pipeline, "plot_main_insights", lambda _df: [
            f"outputs/figures/fig_0{i}_test.png" for i in range(1, 7)
        ],
    )
    monkeypatch.setattr(
        pipeline, "_load_yahya_modules", lambda: None,
    )

    result = pipeline.run_pipeline()

    manifest = json.loads((tmp_path / "outputs/reports/run_manifest.json").read_text())
    assert result["status"] == "awaiting_yahya"
    assert manifest["status"] == "awaiting_yahya"
    assert manifest["completed_steps"] == [
        "official_cleaning_and_quality_report",
        "canonical_load_and_validation",
        "descriptive_transformation",
        "descriptive_statistics",
        "amirul_descriptive_figures",
    ]
    assert manifest["outputs"]["tables"]["summary_statistics"] == "outputs/tables/summary_statistics.csv"
    assert set(manifest["pending_steps"]) == {
        "engineer_features", "train_failure_model", "predict_one", "plot_failure_insights"
    }
