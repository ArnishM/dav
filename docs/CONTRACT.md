# Industrial Machine Failure Analyzer — Shared Contract

Schema version: `1.0`. This is the single authoritative shared integration contract. The preserved Yahya handoff is background documentation; its references to `docs/contracts.md` refer to this file. No second contract file should be created. User-authorized Phase 1 decisions below define Yahya's interfaces; proposals for Amirul's unresolved interfaces are explicitly marked pending approval.

## Scope and ownership

Analyze synthetic machinery observations through preparation, statistics, transformations, charts and snapshot failure classification. The source has no genuine timestamps: do not claim real-time readings, chronological forecasting or remaining useful life. Preserve raw data and physical units; meaningful sensor outliers must not be automatically removed.

Yahya owns `features.py`, `prediction.py`, `advanced_analysis.py` under `src/industrial_analyzer/`, `notebooks/03_failure_insights.ipynb` and `tests/test_prediction.py`. Yahya fixes defects in these modules.

Amirul owns data loading and validation, cleaning and preprocessing, transformations and aggregation, statistics, main DAV charts, final notebook and pipeline orchestration. Planned modules are `data_loader.py`, `validation.py`, `cleaning.py`, `transformation.py`, `statistics.py`, `visualization.py`; integration files are `scripts/run_pipeline.py` and `notebooks/04_final_demo.ipynb`. Do not implement or edit his modules without his request. Shared documentation, dependencies, package initialization and contract tests require coordination.

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

These signatures are documentation, not implemented code. `pd` denotes `pandas`.

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

For initial development, run from the repository root with `src` on `PYTHONPATH`; avoid notebook-specific path hacks. This Phase 1 scaffold has no Python package implementation yet. Python version and dependency pins must be agreed and verified before installation.

## Compatibility tests planned for coding phases

Shared `tests/test_contract.py` will check CSV column order, dtypes/domains, unique IDs, required values, raw units, public signatures and JSON structures. Coordinate ownership before editing it. Yahya's tests will verify non-mutating feature calculations and row preservation, exactly six model inputs, no label leakage, training-only fitted preprocessing, deterministic split/training, serialization round-trip, native JSON types, invalid-input exceptions and chart paths/nonempty PNGs. Use clearly labeled small artificial fixtures, not invented source observations or claimed performance.

Integration smoke tests will load Amirul's canonical CSV and invoke all four public functions without notebook dependencies. Check returned paths exist, confusion counts and metrics are consistent, positive-class probabilities use the correct class, and names do not collide. These are future checks; no tests or model results exist in Phase 1.

## Proposals pending Amirul's approval

These are NOT established interfaces for his modules:

- Agree on his public loading/validation/cleaning/transformation/statistics function names and signatures; none are prescribed here.
- Propose `data_quality.json` include schema version, source path, before/after row counts, missing-value counts, duplicate-ID counts, outlier flags and logged cleaning actions. Exact keys and types require his approval.
- Propose `run_manifest.json` include schema version, input paths, output paths, pipeline completion status and errors. Exact keys/types and partial-failure behavior require his approval.
- Agree on transformed CSV and table column schemas and six core figure filenames. Preserve target labels and meaningful outliers; demonstrate artificial dirt in a separately labeled copy.
- Agree on a supported Python version, tested dependency pins, and whether small generated evidence files will be intentionally committed. Agree on shared scaffold integration into main only after explicit commit/push/merge authorization.

Changes to shared schema, interfaces, ownership or paths require both students' agreement and an updated contract. Preserve the original handoff unchanged.
