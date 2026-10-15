# Yahya integration handoff — Industrial Machine Failure Analyzer

Prepared for Amirul and future assistants, 10 October 2026. `docs/CONTRACT.md` remains authoritative; this document explains the existing implementation without changing shared requirements.

## 1. Purpose and completion status

The DAV project studies machinery observations using data preparation, statistics, transformations, visualization and snapshot machine-failure classification. AI4I 2020 is a synthetic dataset without genuine timestamps. It does not establish real-time monitoring, chronological forecasting, remaining useful life or certified equipment safety diagnoses. Comparisons in the charts are descriptive, not causal.

Yahya's Phases 2–5 implement and test feature engineering, Decision Tree prediction, three failure-focused charts and their integration. The published implementation baseline is commit `19aed6d1652fbb04283e14f7d81353f207a75c7b` on `yahya`. The complete group project is not finished: Amirul's canonical cleaned CSV, orchestration and final notebook integration remain pending.

Earlier official-source smoke tests used an in-memory mapping of the original UCI CSV, not Amirul's permanent cleaning module. The previously verified source has 10,000 observations, 14 columns and 339 failure labels. Earlier prediction and visualization smoke tests completed; those checks do not prove compatibility with Amirul's future cleaned CSV. This handoff does not introduce new performance claims or reproduce unavailable evaluation artifacts. Phase 5 end-to-end checks use explicitly artificial records. Their metrics are test-only, not project performance results.

## 2. Ownership and existing files

| Existing file | Purpose |
|---|---|
| `src/industrial_analyzer/features.py` | Canonical validation and descriptive feature engineering |
| `src/industrial_analyzer/prediction.py` | Training, evaluation, pipeline serialization and single-record prediction |
| `src/industrial_analyzer/advanced_analysis.py` | Failure-focused PNG charts |
| `tests/test_features.py` | Feature and validation unit tests |
| `tests/test_prediction.py` | Prediction unit tests |
| `tests/test_advanced_analysis.py` | Chart unit tests |
| `tests/test_integration_yahya.py` | Connections among all four functions and optional cleaned-CSV compatibility |
| `docs/Yahya_Integration_Handoff.md` | This usage guide |

Yahya maintains his implementation modules and tests. Shared documentation, dependencies and contract tests require coordination. The contract assigns notebook `03_failure_insights.ipynb` to Yahya, but this handoff does not claim that notebook is implemented.

Amirul owns loading, validation, cleaning, transformations, aggregation, statistics, main DAV charts, orchestration and the final notebook. `scripts/run_pipeline.py` and `notebooks/04_final_demo.ipynb` are planned integration locations, not executable interfaces supplied by Yahya. Import the public functions; do not copy notebook cells or edit their internal implementation.

## 3. Dataset Amirul must provide

Write `data/processed/ai4i_cleaned.csv`: UTF-8, comma-separated, one header row, no saved DataFrame index. It is the common source of truth. The exact 14-column order is:

```csv
uid,product_id,type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_failure,twf,hdf,pwf,osf,rnf
```

| Field | Required values and units |
|---|---|
| `uid` | Unique integer identifier |
| `product_id` | Nonempty string identifier |
| `type` | Exactly `L`, `M` or `H` |
| `air_temperature_k`, `process_temperature_k` | Finite numbers greater than zero, Kelvin |
| `rotational_speed_rpm` | Finite number greater than zero, RPM |
| `torque_nm` | Finite number zero or greater, Nm |
| `tool_wear_min` | Finite number zero or greater, minutes |
| `machine_failure` | Integer 0 or 1; classification target, positive class 1 |
| `twf`, `hdf`, `pwf`, `osf`, `rnf` | Integer 0 or 1; descriptive failure-mode labels |

No missing values, NaN, infinity, duplicate column names or duplicate IDs are accepted. Numeric strings and booleans are invalid numerical inputs; float labels such as `1.0` are not accepted as integer labels. Preserve physical units and meaningful sensor extremes. Do not rewrite failure labels from assumed relationships among flags.

The original UCI header uses `UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`. Amirul's loader must map these to the canonical names and order without changing the original CSV. Yahya's public functions do not perform that mapping or clean invalid data.

`data/processed/ai4i_transformed.csv` is only for DAV demonstrations. Never pass globally scaled/encoded transformed data to training.

## 4. Interfaces and the 14-versus-16-column rule

```python
engineer_features(df: pd.DataFrame) -> pd.DataFrame
train_failure_model(df: pd.DataFrame, model_path: str) -> dict
predict_one(readings: dict, model_path: str) -> dict
plot_failure_insights(df: pd.DataFrame, output_dir: str) -> list[str]
```

`engineer_features` accepts exactly the canonical 14 columns. It returns a separate DataFrame preserving original values, dtypes, index and row/column order, followed by:

- `temperature_difference_k = process_temperature_k - air_temperature_k`.
- `torque_speed_product = torque_nm * rotational_speed_rpm` (Nm × RPM, not watts).

Training also accepts exactly 14 columns. It internally selects only `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. Identifiers, target, failure flags and both derived columns are excluded. Passing the 16-column engineered result to training is an error.

Plotting accepts the canonical 14 columns alone or those same columns followed by both derived columns in the order above. A partial pair, extra columns, nonfinite derived values or calculations inconsistent beyond the implementation's `1e-9` relative/absolute tolerance are rejected. All public functions preserve the caller's DataFrame.

## 5. Environment and imports

Use the existing project-local `.venv`. The environment used for current checks runs Python 3.12.10. `requirements.txt` lists pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, pytest, jupyterlab and ipykernel; versions are currently unpinned. PNG readability tests also import `PIL.Image` from Pillow, currently installed through Matplotlib's dependencies. Agree on a reproducible team environment before changing dependencies; do not assume every fresh environment matches this one.

From PowerShell, start in the repository root:

```powershell
Set-Location 'C:\Users\LENOVO\Industrial Machine Failure Analyzer'
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
```

`PYTHONPATH` allows Python to find the package in `src`; it is not a package installation. For a terminal started in a nested folder, set `PYTHONPATH` to the absolute repository `src` path instead. A Jupyter kernel should be launched from this configured environment; restart the kernel after environment changes. `Agg` saves images without requiring a display window.

## 6. Complete minimal usage example

Run this example from the repository root using the configured environment, after Amirul supplies the canonical CSV. It is example code for his integration, not a newly created orchestrator. It writes the standard model, report and three charts, replacing previous outputs at those same Yahya-owned paths.

```python
from pathlib import Path
import pandas as pd

from industrial_analyzer.features import engineer_features
from industrial_analyzer.prediction import train_failure_model, predict_one
from industrial_analyzer.advanced_analysis import plot_failure_insights

repository = Path.cwd()  # Run this example from the repository root.
clean_df = pd.read_csv(
    repository / "data/processed/ai4i_cleaned.csv", encoding="utf-8"
)

features = engineer_features(clean_df)
model_info = train_failure_model(clean_df, "models/failure_model.joblib")

# Illustrative readings, not a claimed official dataset observation.
readings = {
    "type": "L",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}
prediction = predict_one(readings, model_info["model_path"])
figure_paths = plot_failure_insights(features, "outputs/figures")

print(model_info)
print(prediction)
for relative_path in figure_paths:
    print(repository / relative_path)
```

The module output paths resolve from the repository containing the source files (`Path(__file__).resolve().parents[2]`), independent of the caller's working directory. The example's CSV read uses its explicitly stated root assumption. Amirul's orchestrator should discover its repository root explicitly for CSV loading too. Keep the source layout intact; this is not yet an installed-package deployment design.

## 7. Training and prediction behavior

Training requires both classes and enough observations to retain both in the 80/20 stratified split and training validation folds. Seed 42 fixes split and tree randomness. A Pipeline contains a ColumnTransformer that one-hot encodes `type` and passes through five numeric inputs, followed by a DecisionTreeClassifier with balanced class weights. No scaling is used. Depths 3, 5 and 8 are compared by F1 using training-only StratifiedKFold validation (up to three folds, at least two). Preprocessing fits inside each training fold; the held-out test set never selects depth. The best pipeline is refitted on the training partition and evaluated on held-out data; it is not subsequently refitted on all observations.

Training returns exactly `schema_version`, `model_path`, `metrics`. The entire result is also saved as strict JSON, without NaN/Infinity:

```text
schema_version: "1.0"
model_path: repository-relative string
metrics:
  model_name: "decision_tree"
  evaluation_set: "held_out_test"
  positive_class: 1
  random_state: 42
  test_size: 0.2
  train_samples, test_samples, failure_support: integers
  accuracy, precision, recall, f1: numbers in [0, 1]
  confusion_matrix: [[TN, FP], [FN, TP]]
```

Confusion rows are actual classes and columns predicted classes, ordered `[0, 1]`. Counts sum to `test_samples`; `FN + TP` equals `failure_support`. Precision, recall and F1 use positive class 1 and zero-division fallback 0. Rare failures make accuracy alone misleading: inspect failure recall, precision, F1 and the matrix too.

`predict_one` accepts exactly the six keys in the example, validates their categories and physical domains, loads a trusted fitted pipeline and never retrains. The probability column is located from classifier class label 1, not an assumed position. Its exact return keys are:

```text
schema_version: "1.0"
predicted_failure: native Python int, 0 or 1
failure_probability: native Python float in [0, 1], or None
model_name: "decision_tree"
message: "Model classification; not a certified equipment diagnosis"
```

Probability is a model estimate, not guaranteed empirical risk. Only load trusted project-generated Joblib files.

## 8. Outputs and coexistence with Amirul's analytics

| Standard output | Meaning |
|---|---|
| `models/failure_model.joblib` | Complete fitted preprocessing/classifier pipeline |
| `outputs/reports/model_metrics.json` | Exact training-return metadata and evaluation metrics |
| `outputs/figures/fig_y_01_torque_by_failure.png` | Box plot: recorded failure groups on x, torque in Nm on y, group sample counts |
| `outputs/figures/fig_y_02_torque_vs_rpm.png` | Scatter: RPM on x, torque in Nm on y; color/legend for recorded failure status |
| `outputs/figures/fig_y_03_failure_rate_by_type.png` | Bars: L/M/H on x, failures divided by all records of that type × 100 on y; sample counts |

Failure points are drawn after non-failures in the scatter. No observations are artificially balanced. Existing types with zero failures show 0%; absent types show N/A with n=0. Figures are closed after saving. Returned chart paths are three forward-slash repository-relative strings.

Relative and repository-contained absolute output paths are supported; escaping paths and empty strings are rejected. Parent folders are created as needed. Changing `model_path` does NOT redirect the report: training always writes `outputs/reports/model_metrics.json`. Deterministic reruns replace the function's own files. Avoid concurrent training runs sharing this report destination. There is no transactional rollback guarantee if a later filesystem operation fails.

Amirul's `fig_*.png` names must exclude the reserved `fig_y_` subset. His processed CSVs, data quality report, manifest and tables remain separate. The four functions do not implement his loader, transformation, statistics or orchestration and do not create his outputs.

## 9. Tests and previously skipped CSV check

From the repository root with the environment settings above:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_prediction.py tests/test_advanced_analysis.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_integration_yahya.py -v
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

The integration suite uses a 60-row artificial fixture, runs all four functions, reloads the pipeline, checks exactly six inputs, validates report/count consistency and readable PNGs, preserves input/sentinel files, and repeats execution from ordinary and nested working directories. Error tests check single-class failure propagation and rejection of engineered training data. Both prediction and visualization repository roots are monkeypatched to the same pytest temporary directory, including the fixed metrics destination. Permanent project artifacts are not overwritten.

Handoff verification on 10 October 2026 ran the full `tests/ -q` command above: **148 passed, 0 failed, 1 skipped in 15.62 seconds**. The skip was the absent Amirul cleaned CSV. This records the current execution, not a guarantee for later combined branches or environments.

When Amirul supplies his real canonical file at `data/processed/ai4i_cleaned.csv`, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_integration_yahya.py::test_amirul_cleaned_csv_read_only -v -rs
```

This test automatically loads that file, checks canonical column order and numeric/integer dtypes, invokes the four functions with isolated outputs, checks DataFrame preservation and verifies the CSV's SHA-256 remains unchanged. If the file is absent, it deliberately skips with a clear message. Do not generate a replacement to remove the skip. Rerun the whole suite after compatibility succeeds.

## 10. Common errors and solutions

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: industrial_analyzer` | Use the project `.venv`; set `PYTHONPATH=src` from root or an absolute src path from nested folders |
| Missing dependency | Check the selected Python environment and agreed requirements; obtain authorization before installation |
| Expected 14 canonical columns / derived collision | Pass `clean_df` to training/features; fix loader names/order; omit saved index columns |
| Invalid integers, categories, missing or nonfinite values | Correct Amirul's upstream validation/cleaning; do not silently invent values or erase meaningful outliers |
| Both classes / insufficient observations | Supply enough valid examples of both outcomes for split and validation; do not duplicate or fabricate records merely to train |
| Partial/inconsistent derived columns | Pass canonical data directly, or regenerate the complete pair with `engineer_features` |
| Missing model (`FileNotFoundError`) | Train successfully first and pass returned `model_path` |
| Incompatible model (`ValueError`) | Use this project's complete trusted fitted pipeline; do not supply a bare classifier or unrelated artifact |
| Escaping output path (`ValueError`) | Select a location inside the repository |
| Filesystem `OSError` / `PermissionError` | Check permissions, available storage and destinations; report failure rather than fake success |
| Pytest temporary-directory permission restriction | Request a safe permitted rerun; an environment restriction is not a model defect |
| Cleaned-CSV test skipped | Provide Amirul's canonical CSV, then rerun the targeted check |

Exceptions should propagate to Amirul's orchestrator for reporting. A failed run must not be presented as a completed pipeline.

## 11. Team review and integration into main

These are instructions for team review and a later authorized merge. Publishing the handoff or opening a pull request does not authorize merging into main.

1. Preserve local work and inspect status before synchronization. Run `git fetch origin`, then inspect `git log --oneline origin/main..origin/yahya` and `git diff origin/main...origin/yahya`. The published Yahya implementation and handoff are accessible on `origin/yahya` after the authorized documentation push.
2. On GitHub, open a comparison with base `main` and compare `yahya`: `https://github.com/yjd-lab/dav/compare/main...yahya`. Review an existing open pull request if present; otherwise create one only when authorized. This document does not claim an open PR exists.
3. Review code, ownership boundaries, contract compatibility and actual test results. If Amirul needs to test on his branch, he should manage that checkout and integration himself, preserving his uncommitted work. Do not force-push or reset histories to resolve conflicts.
4. Run the cleaned-CSV compatibility test and full suite on the proposed integration state. Resolve genuine conflicts together without changing shared interfaces silently. Merge into `main` only after team review and explicit authorization. Keep `main` as the stable integrated branch.
5. In Amirul's final pipeline, produce/load the canonical cleaned CSV, call the four functions in the example order, then incorporate returned metadata and paths alongside his own statistics, tables and figures. Keep transformed demonstration data out of model training. Verify input hashes, six model features, complete pipeline reloading, report consistency and disjoint output names.
6. Once Amirul implements and documents `scripts/run_pipeline.py`, use its agreed invocation to execute the final project. No final command-line options or callable interface exist in this handoff; do not assume `python scripts/run_pipeline.py` currently works. His final notebook should demonstrate the reusable functions rather than duplicate their internals.

## 12. Pending decisions and acceptance checklist

- Provide and test Amirul's canonical cleaned CSV; confirm raw preservation and cleaning decisions.
- Agree on his public loader/cleaner/transformation/statistics interfaces, table schemas, main figure names, data-quality JSON and run-manifest JSON; the contract labels these proposals pending approval.
- Agree on Python/dependency versions and how generated evidence is retained; model/report files are currently ignored by Git.
- Implement and test his orchestration and final notebook, including useful error reporting and partial-output handling.
- Recheck the complete combined project after branch integration; verify no namespace collisions and no label leakage.
- Review and explicitly authorize documentation publication, any PR and final merge separately. Passing Yahya's tests does not mean the whole group project is complete.
