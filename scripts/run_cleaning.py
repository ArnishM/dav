"""Generate the official cleaned data and a separate dirty-data demo."""

import json
from pathlib import Path

import pandas as pd

from industrial_analyzer.cleaning import clean_dirty_demo_data, clean_official_ai4i_data
from industrial_analyzer.data_loader import load_ai4i_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "data/raw/ai4i2020.csv"
DIRTY_PATH = REPOSITORY_ROOT / "data/demo/ai4i_dirty_demo.csv"
MANIFEST_PATH = REPOSITORY_ROOT / "data/demo/ai4i_dirty_demo_manifest.json"


def create_dirty_demo(source: pd.DataFrame) -> pd.DataFrame:
    """Introduce documented errors in a separate copy of source data."""
    dirty = source.copy(deep=True)

    dirty.loc[dirty["uid"] == 1, "air_temperature_k"] = float("nan")
    dirty.loc[dirty["uid"] == 2, "type"] = None
    dirty.loc[dirty["uid"] == 5, "torque_nm"] = -1

    exact_duplicate = dirty.loc[dirty["uid"] == 3].copy()
    conflicting_record = dirty.loc[dirty["uid"] == 4].copy()
    conflicting_record.loc[:, "torque_nm"] = conflicting_record["torque_nm"] + 1
    dirty = pd.concat([dirty, exact_duplicate, conflicting_record], ignore_index=True)
    return dirty


def main() -> None:
    source = load_ai4i_csv(SOURCE_PATH)
    dirty_demo = create_dirty_demo(source)

    DIRTY_PATH.parent.mkdir(parents=True, exist_ok=True)
    dirty_demo.to_csv(DIRTY_PATH, index=False, encoding="utf-8")

    manifest = {
        "description": "Artificial dirty-data demonstration derived from the official UCI CSV.",
        "source_path": "data/raw/ai4i2020.csv",
        "demo_path": "data/demo/ai4i_dirty_demo.csv",
        "source_modified": False,
        "introduced_cases": [
            {"uid": 1, "column": "air_temperature_k", "change": "set to missing"},
            {"uid": 2, "column": "type", "change": "set to missing"},
            {"uid": 3, "change": "added one exact duplicate row"},
            {
                "uid": 4,
                "column": "torque_nm",
                "change": "added a conflicting record with torque increased by 1 Nm",
            },
            {"uid": 5, "column": "torque_nm", "change": "set to -1 Nm (invalid domain value)"},
        ],
        "identifiers_and_failure_labels_modified": False,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    official = clean_official_ai4i_data()
    demo_clean, demo_quarantine, demo_report = clean_dirty_demo_data(dirty_demo)

    print(f"Official cleaned CSV: {len(official)} rows, {len(official.columns)} columns")
    print(f"Official report: outputs/reports/data_quality.json")
    print(f"Dirty demo source: {len(dirty_demo)} rows")
    print(f"Dirty demo cleaned: {len(demo_clean)} rows")
    print(f"Dirty demo quarantined: {len(demo_quarantine)} rows")
    print(f"Dirty demo report: outputs/reports/dirty_demo_quality.json")
    print(f"Dirty demo manifest: data/demo/ai4i_dirty_demo_manifest.json")
    print(f"Dirty demo outlier readings flagged: {demo_report['actions']['flagged_outliers']}")


if __name__ == "__main__":
    main()
