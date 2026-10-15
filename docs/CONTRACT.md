# Industrial Machine Failure Analyzer — Shared Contract

Schema version: `1.0`. This is the single authoritative shared integration contract. The preserved Yahya handoff is background documentation; its references to `docs/contracts.md` refer to this file. No second contract file should be created. The project implementation and integration are present on the `amirul` branch. Exact transformed CSV, summary-table, quality-report and run-manifest schemas remain pending joint confirmation; this document does not treat the current output shapes as approved until that review is complete.

## Scope and ownership

Analyze synthetic machinery observations through preparation, statistics, transformations, charts and snapshot failure classification. The source has no genuine timestamps: do not claim real-time readings, chronological forecasting or remaining useful life. Preserve raw data and physical units; meaningful sensor outliers must not be automatically removed.

Yahya owns `features.py`, `prediction.py`, `advanced_analysis.py` under `src/industrial_analyzer/` and `tests/test_prediction.py`. Yahya fixes defects in these modules. By team agreement, a separate `notebooks/03_failure_insights.ipynb` is not required: `notebooks/04_final_demo.ipynb` runs the integrated pipeline and demonstrates feature engineering, model evaluation, the six-field prediction example and Yahya's advanced charts.

Amirul owns data loading and validation, cleaning and preprocessing, transformations and aggregation, statistics, main DAV charts, final notebook and pipeline orchestration. Implemented modules are `data_loader.py`, `validation.py`, `cleaning.py`, `transformation.py`, `statistics.py` and `visualization.py`; orchestration and demonstration are in `scripts/run_pipeline.py` and `notebooks/04_final_demo.ipynb`. Do not implement or edit his modules without his request. Shared documentation, dependencies, package initialization and contract tests require coordination.

## Canonical cleaned CSV

Common source of truth: `data/processed/ai4i_cleaned.csv`. UTF-8, comma-separated, exactly one header row, no DataFrame index or `Unnamed: 0` column. Exactly these 14 columns in this order:

```csv
uid,product_id,type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_failure,twf,hdf,pwf,osf,rnf
```

| Column | Type and validation | Units / role |
|---|---|---|
| `uid` | Unique integer | Identifier; generally sorted ascending, never a timestamp |
| `product_id` | Nonempty string | Identifier |
| `type` | String, exactly `L`, `M` or `H` | Product category |
| `air_temperature_k` | Finite number, greater than zero | Kelvin |
| `process_temperature_k` | Finite number, greater than zero | Kelvin |
| `rotational_speed_rpm` | Finite number, greater than zero | RPM |
| `torque_nm` | Finite number, zero or greater | Nm |
| `tool_wear_min` | Finite number, zero or greater | Minutes |
| `machine_failure` | Integer `0` or `1` | Target |
| `twf` | Integer `0` or `1` | Descriptive failure flag |
| `hdf` | Integer `0` or `1` | Descriptive failure flag |
| `pwf` | Integer `0` or `1` | Descriptive failure flag |
| `osf` | Integer `0` or `1` | Descriptive failure flag |
| `rnf` | Integer `0` or `1` | Descriptive failure flag |

No required value may be missing. Reject NaN, infinity and duplicate column names. Boolean values are not accepted as numbers or binary labels. Do not impose arbitrary upper bounds, trim meaningful extremes, or require a particular temperature difference. These domain checks validate this operating-machine dataset rather than establishing equipment safety limits. Do not rewrite labels based on an assumed relationship among failure flags.

`data/processed/ai4i_transformed.csv` is for DAV demonstrations only. It is not a model input: global scaling or encoding may leak information into evaluation.

## Cleaning and data-quality policies

- Keep `data/raw/ai4i2020.csv` unchanged. The official cleaned CSV keeps exactly the 14 canonical columns and preserves the original physical sensor values.
- For official data, detect and report missing values, exact duplicate rows, duplicate `uid` values, invalid records and IQR outliers. Do not impute or remove official observations. If unexpected invalid data is found, write the report and stop before producing the cleaned CSV.
- Use the IQR rule with fences at 1.5 times the interquartile range to flag unusual numeric sensor readings. Report them without removing or altering them.
- Keep deliberate dirty-data examples under `data/demo/`, derived from a separate copy of the official source. Record each intentional change in a manifest.
- In a dirty demonstration, remove and count exact duplicate rows; flag and quarantine conflicting records with the same `uid`; median-impute missing numeric sensor values; and mode-impute missing `type` values. Never invent identifiers or failure labels. Quarantine invalid or ambiguous records and document all corrections, removals, quarantines and flags.
- The official quality report is `outputs/reports/data_quality.json`. Dirty-demonstration quality results are reported separately from official data.

## Prediction features

Ordered allowlist: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. Target: `machine_failure`; positive class is `1` (failure), negative class is `0`.

Exclude `uid`, `product_id`, `machine_failure`, `twf`, `hdf`, `pwf`, `osf`, `rnf` from predictors. Exclude derived descriptive columns from the initial model. Always select the allowlist explicitly rather than dropping only the target.

## Yahya's public interfaces

These are the implemented public interfaces. `pd` denotes `pandas`.

```python
engineer_features(df: pd.DataFrame) -> pd.DataFrame
train_failure_model(df: pd.DataFrame, model_path: str) -> dict
predict_one(readings: dict, model_path: str) -> dict
plot_failure_insights(df: pd.DataFrame, output_dir: str) -> list[str]
```

### engineer_features — features.py, Yahya

Accept a nonempty canonical cleaned DataFrame. Validate the canonical schema and values. Return a NEW DataFrame preserving its index, row order, original column order and values, without mutating the argument. Append:

- `temperature_difference_k = process_temperature_k - air_temperature_k` (Kelvin difference).
- `torque_speed_product = torque_nm * rotational_speed_rpm` (Nm × RPM; proportional load proxy, not watts).

Reject existing derived-name collisions or nonfinite calculated results with `ValueError`. These columns initially serve descriptive analytics only.

### train_failure_model — prediction.py, Yahya

Accept a nonempty canonical cleaned DataFrame and nonempty model-path string. Reject invalid schema, single-class targets or insufficient class counts for stratification with a helpful `ValueError`. Do not mutate the input. Require both classes in training and test partitions.

Use `test_size=0.2`, `stratify=y`, `random_state=42`. Fit preprocessing only on training data, including within training-only validation folds. Use a saved scikit-learn Pipeline with a ColumnTransformer: one-hot encode `type`, pass through the five numeric features, and fit an initial Decision Tree with seed 42 and balanced class weights. Tree depth is an implementation choice selected using training-only validation; never tune on the held-out test set. Scaling is unnecessary for the initial tree.

Save the entire fitted preprocessing/classifier pipeline, not just the tree. Standard call uses `model_path="models/failure_model.joblib"`. The argument can select another repository-contained location for tests; metadata must report the actual location. Always write evaluation metadata to `outputs/reports/model_metrics.json`. Return exactly `schema_version`, `model_path`, `metrics`, using the same object structure as that JSON file.

Evaluate the saved model on the held-out test partition; do not refit it on all observations while reporting the earlier test metrics. All numeric outputs must be finite native Python scalars. No invented results or placeholder metrics may be written.

### Model metrics JSON structure

Required structure below is a type specification, not sample performance:

```text
{
  "schema_version": "1.0",
  "model_path": string (repository-relative path),
  "metrics": {
    "model_name": "decision_tree",
    "evaluation_set": "held_out_test",
    "positive_class": 1,
    "random_state": 42,
    "test_size": 0.2,
    "train_samples": integer,
    "test_samples": integer,
    "failure_support": integer,
    "accuracy": number in [0, 1],
    "precision": number in [0, 1],
    "recall": number in [0, 1],
    "f1": number in [0, 1],
    "confusion_matrix": [[integer TN, integer FP], [integer FN, integer TP]]
  }
}
```

Precision, recall and F1 use binary averaging with positive label 1 and `zero_division=0`. Accuracy is the fraction correctly classified; precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is their harmonic mean. Confusion-matrix labels are explicitly `[0, 1]`; rows are actual classes and columns predicted classes. `failure_support = FN + TP`; all four counts sum to `test_samples`. Accuracy alone is inadequate for rare failures. A no-failure baseline comparison is optional and needs an agreed extension before adding JSON fields. Serialize strict JSON without NaN/Infinity.

### predict_one — prediction.py, Yahya

Accept a dictionary with EXACTLY six keys. Dictionary insertion order is irrelevant; select the allowlist order internally. Example inputs below are illustrative, not a claimed source observation:

```json
{
  "type": "L",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 0
}
```

Apply the same category, finite numeric and physical-domain checks as the CSV. Reject missing/extra keys, numeric strings, booleans, nulls, unknown categories, NaN and infinity with field-specific `ValueError`. Do not silently fill defaults. Load the saved pipeline and use its fitted preprocessing.

Return exactly:

| Key | Value |
|---|---|
| `schema_version` | String `"1.0"` |
| `predicted_failure` | Python integer `0` or `1` |
| `failure_probability` | Python float in `[0, 1]` for class `1` when valid predict_proba is available; otherwise `None` |
| `model_name` | String `"decision_tree"` |
| `message` | String `"Model classification; not a certified equipment diagnosis"` |

Locate the probability column using classifier class labels, not a guessed column position. Reject invalid model predictions/probabilities with `ValueError`. A probability is a model estimate, not guaranteed empirical risk. Model artifacts must come from trusted project runs.

### plot_failure_insights — advanced_analysis.py, Yahya

Accept a nonempty canonical DataFrame, optionally with the two derived columns appended, and a nonempty output-directory string. Validate canonical fields and any derived fields used. Return `list[str]` of paths to at least two nonempty PNG files. Preserve the input. Use descriptive failure-conditioned charts with units, labels and readable legends.

Names begin `fig_y_`, for example `fig_y_01_sensor_by_failure.png` and `fig_y_02_feature_relation.png`. Amirul's `fig_` namespace must exclude the reserved `fig_y_` subset. Deterministic reruns may replace the same function's own generated charts, never another owner's files.

This function must not require model/test predictions absent from its arguments. A confusion-matrix chart, if added, belongs to training and may only use actual held-out predictions; its filename is `fig_y_03_model_confusion_matrix.png`.

## Paths and errors

All relative paths resolve from the repository root, not notebook location or arbitrary working directory. Public functions return repository-relative paths using forward slashes. Repository-contained absolute paths may be accepted and normalized; reject empty paths or destinations escaping the repository with `ValueError`. Generated-output parent directories may be created by the owning function. Document root discovery during implementation; tests may use a temporary test repository root.

Invalid caller data, schema, domain values, derived-name collisions or incompatible model contents raise descriptive `ValueError` identifying the problem. Missing model files raise `FileNotFoundError`. Filesystem failures retain appropriate `OSError`/`PermissionError`; do not turn failures into successful-looking outputs. Missing inputs must not trigger automatic dataset downloads. Exceptions propagate to the orchestrator with useful context. The orchestrator should report failure, not fabricate a substitute result.

## Agreed outputs and connection

| Owner | Path / pattern |
|---|---|
| Amirul | `data/processed/ai4i_cleaned.csv` |
| Amirul | `data/processed/ai4i_transformed.csv` |
| Amirul | `outputs/reports/data_quality.json` |
| Amirul | `outputs/reports/run_manifest.json` |
| Amirul | `outputs/tables/summary_statistics.csv` |
| Amirul | `outputs/tables/failure_by_type.csv` |
| Amirul | `outputs/tables/failure_modes.csv` |
| Amirul | `outputs/figures/fig_*.png` excluding `fig_y_*.png` |
| Yahya | `models/failure_model.joblib` |
| Yahya | `outputs/reports/model_metrics.json` |
| Yahya | `outputs/figures/fig_y_*.png` |

Amirul loads, validates and cleans the data, then writes the canonical CSV. His orchestrator loads that CSV, calls `engineer_features(clean)` for descriptive analysis, calls `train_failure_model(clean, "models/failure_model.joblib")`, calls `predict_one(readings, model_info["model_path"])`, and calls `plot_failure_insights(features, "outputs/figures")`. No REST server is required. He imports Yahya's functions; he must not copy notebook cells or rewrite their internals.

Run commands from the repository root with `src` on `PYTHONPATH`; notebooks also expect the repository root as the working directory. The agreed development interpreter is Python 3.13.11 in the project-specific `.venv`. Dependencies are listed in `requirements.txt`; package versions are not pinned.

## Test and compatibility status

The integrated test suite was run on the `amirul` branch during the 2026-10-15 pre-merge audit: 186 passed, 0 failed, 0 skipped. The compatibility test reads `data/processed/ai4i_cleaned.csv`; it skips only when that file is absent. Its current test name is `test_amirul_cleaned_csv_read_only` in `tests/test_integration_yahya.py`.

The Yahya unit and integration tests cover non-mutating feature calculations and row preservation, exactly six model inputs, no label leakage, training-only preprocessing, deterministic split/training, serialization, metrics, invalid-input handling, chart paths and output-name separation. `tests/test_integration_yahya.py` exercises the canonical cleaned CSV when present. The passing test suite verifies implementation behavior; the schemas below are the separately agreed output contract.

## Agreed descriptive output schemas

Yahya and Amirul approved the following schemas for the descriptive outputs. CSV files are UTF-8, comma-separated, have one header row, and do not include a DataFrame index.

### Transformed demonstration CSV

`data/processed/ai4i_transformed.csv` has these 28 columns in exactly this order:

```csv
uid,product_id,type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_failure,twf,hdf,pwf,osf,rnf,type_L,type_M,type_H,air_temperature_k_minmax,air_temperature_k_zscore,process_temperature_k_minmax,process_temperature_k_zscore,rotational_speed_rpm_minmax,rotational_speed_rpm_zscore,torque_nm_minmax,torque_nm_zscore,tool_wear_min_minmax,tool_wear_min_zscore,tool_wear_quantile_bin
```

The first 14 columns retain the canonical values and order. Added columns are descriptive encodings, global min-max/z-score transformations and a tool-wear quantile bin. This file is for demonstrations only and must never be used as model input.

### Summary tables

| File | Exact columns, in order |
|---|---|
| `outputs/tables/summary_statistics.csv` | `feature, count, mean, std, min, 25%, 50%, 75%, max` |
| `outputs/tables/failure_by_type.csv` | `type, total_records, failure_count, nonfailure_count, failure_rate_percent` |
| `outputs/tables/failure_modes.csv` | `failure_mode, flagged_records, percent_of_records, dataset_records` |

### Official data-quality report

For a successfully parsed official source, `outputs/reports/data_quality.json` has exactly these top-level keys:

```text
schema_version, dataset_kind, source_path, cleaned_path, status, row_counts,
missing_values_by_column, duplicates, validation_errors, outliers, actions,
source_preserved
```

The nested keys and value types are:

- `schema_version`, `dataset_kind`, `source_path`, `cleaned_path`, `status`: strings; `status` is `success` or `failed`.
- `row_counts`: `input` (integer), `output` (integer or null when validation failed), `removed` (integer).
- `missing_values_by_column`: object mapping each canonical column name to an integer missing-value count.
- `duplicates`: `exact_duplicate_rows`, `duplicate_uid_groups`, `duplicate_uid_records` (integers).
- `validation_errors`: array of strings.
- `outliers`: `method` (string) and `fields` (object keyed by the five sensor measurement columns). Each available field contains `status`, `method`, `lower_fence`, `upper_fence`, `flagged_count`, `uids`; fences are finite numbers and counts are integers. An unavailable field contains `status`, `flagged_count`, `uids`.
- `actions`: `corrected_values`, `removed_rows`, `flagged_outliers` (integers) and `details` (array of strings).
- `source_preserved`: boolean.

If the source cannot be parsed before an audit can be completed, the early-error report has exactly `schema_version, dataset_kind, source_path, status, validation_errors, actions`; its `actions` object has `corrected_values, removed_rows, flagged_outliers`, all integers. The pipeline records the failure in its run manifest and stops.

### Pipeline run manifest

`outputs/reports/run_manifest.json` has exactly these top-level keys:

```text
schema_version, status, inputs, outputs, completed_steps, pending_steps, errors
```

- `schema_version` and `status` are strings. Status is `running`, `complete`, `awaiting_readings`, `awaiting_yahya`, or `failed`.
- `inputs` has `source_csv` (repository-relative path string).
- `outputs` contains only outputs completed by the run. In a complete run with a prediction, its exact keys are `cleaned_csv, data_quality_report, transformed_csv, tables, amirul_figures, model, model_metrics, yahya_figures, single_reading_prediction`. Paths are repository-relative strings; `tables` has exactly `summary_statistics, failure_by_type, failure_modes`; chart lists contain repository-relative path strings; `single_reading_prediction` is `returned_to_caller` because the prediction object is returned to the caller rather than written as a file.
- `completed_steps` and `pending_steps` are arrays of strings.
- `errors` is an array of objects. A caught pipeline exception records `step`, `error_type`, and `message`; a missing Yahya module records `step` and `message`.

For incomplete runs, `outputs`, `completed_steps`, `pending_steps`, and `errors` reflect completed work and the failure or waiting condition; they do not claim unfinished artifacts were generated. The final notebook demonstrates feature engineering, model evaluation, a six-field prediction example and Yahya's advanced charts. By team agreement, no separate `notebooks/03_failure_insights.ipynb` is required.

Generated evidence remains untracked during development. The team will include the necessary reproducible outputs in the university submission; regeneration steps are documented below.

Changes to shared schema, interfaces, ownership or paths require both students' agreement and an updated contract. Preserve the original handoff unchanged.

## Reproducing generated outputs

Run from the repository root with the agreed `.venv` activated. These commands read the official source and write the cleaned and generated files to the paths listed above; they do not edit `data/raw/ai4i2020.csv`.

```bash
cd ~/Documents/dav
source .venv/bin/activate
export PYTHONPATH=src
export MPLBACKEND=Agg
python scripts/run_cleaning.py
```

`run_cleaning.py` regenerates the official clean CSV and quality report, plus the separate dirty demonstration source, its manifest, cleaned result and quarantine CSV. It also writes the dirty-demo quality report under `outputs/reports/`.

To regenerate the transformed CSV, summary tables, model, model metrics, charts and a prediction, provide a JSON file containing exactly the six approved inputs. The following values are illustrative example inputs, not a claimed observation or expected prediction:

```bash
cat > /tmp/dav-readings.json <<'JSON'
{"type":"L","air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":120}
JSON
python scripts/run_pipeline.py --readings-file /tmp/dav-readings.json
```

The final demonstration notebook also regenerates the integrated pipeline outputs when executed from the repository root. The generated CSV/JSON evidence files remain untracked during development; include the agreed final artifacts when packaging the university submission.
