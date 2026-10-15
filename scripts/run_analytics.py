"""Generate the transformed demonstration CSV and descriptive tables."""

from pathlib import Path

from industrial_analyzer.data_loader import load_ai4i_csv
from industrial_analyzer.statistics import write_statistics_tables
from industrial_analyzer.transformation import write_transformed_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cleaned_path = REPOSITORY_ROOT / "data/processed/ai4i_cleaned.csv"
    clean_data = load_ai4i_csv(cleaned_path)
    transformed = write_transformed_csv(clean_data)
    tables = write_statistics_tables(clean_data)

    print(f"Transformed demonstration data: {len(transformed)} rows")
    for name, path in tables.items():
        print(f"{name}: {path.relative_to(REPOSITORY_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
