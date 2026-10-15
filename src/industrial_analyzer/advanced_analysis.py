"""Descriptive failure charts from canonical machinery observations."""

from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from industrial_analyzer.features import (
    CANONICAL_COLUMNS, DERIVED_COLUMNS, _validate_canonical, engineer_features,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIGURE_NAMES = (
    "fig_y_01_torque_by_failure.png",
    "fig_y_02_torque_vs_rpm.png",
    "fig_y_03_failure_rate_by_type.png",
)
GROUP_COLORS = {0: "#3477A5", 1: "#C4483B"}
GROUP_LABELS = {0: "No failure", 1: "Failure"}


def _validate_data(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("df contains duplicate column names")
    present = [name for name in DERIVED_COLUMNS if name in df.columns]
    if present and len(present) != 2:
        raise ValueError("Both derived columns must be present together, or neither")
    expected_columns = list(CANONICAL_COLUMNS) + (list(DERIVED_COLUMNS) if present else [])
    if list(df.columns) != expected_columns:
        raise ValueError("Expected canonical columns in contract order, optionally followed by both derived columns")
    canonical = df.loc[:, list(CANONICAL_COLUMNS)]
    _validate_canonical(canonical)
    if present:
        expected = engineer_features(canonical)
        for name in DERIVED_COLUMNS:
            for value in df[name]:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise ValueError(f"{name} must contain finite real numbers")
            try:
                actual = df[name].to_numpy(dtype=float)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{name} must contain finite real numbers") from error
            if not np.isfinite(actual).all():
                raise ValueError(f"{name} must contain finite real numbers")
            if not np.allclose(actual, expected[name].to_numpy(), rtol=1e-9, atol=1e-9):
                raise ValueError(f"{name} is inconsistent with its documented calculation")


def _output_directory(output_dir: str) -> Path:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a nonempty string")
    root = REPOSITORY_ROOT.resolve()
    directory = Path(output_dir)
    directory = (directory if directory.is_absolute() else root / directory).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("output_dir must be a directory inside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    # Reject links or directories at chart destinations before writing any figure.
    for name in FIGURE_NAMES:
        target = directory / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Chart destination must be a regular file: {target}")
    return directory


def _save_and_close(figure, path: Path) -> None:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=150, format="png")
    except OSError as error:
        raise type(error)(f"Could not save chart to {path}: {error}") from error
    finally:
        plt.close(figure)


def _failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("type", observed=True)["machine_failure"].agg(["sum", "count"])
    summary = summary.reindex(["L", "M", "H"], fill_value=0)
    # Use all records of each type as the denominator, not all failures.
    summary["rate_percent"] = summary["sum"].div(summary["count"].replace(0, np.nan)) * 100
    return summary


def plot_failure_insights(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Validate data and save three descriptive PNGs; return repository-relative paths."""
    _validate_data(df)
    directory = _output_directory(output_dir)

    # Compare torque for recorded failure groups without balancing observations.
    figure, axis = plt.subplots(figsize=(7, 5))
    try:
        present_groups = [label for label in (0, 1) if (df["machine_failure"] == label).any()]
        torque_groups = [df.loc[df["machine_failure"] == label, "torque_nm"].to_numpy(dtype=float) for label in present_groups]
        boxes = axis.boxplot(torque_groups, positions=present_groups, widths=0.5, patch_artist=True)
        for box, label in zip(boxes["boxes"], present_groups):
            box.set_facecolor(GROUP_COLORS[label])
        axis.set_xlim(-0.5, 1.5)
        sns.despine(ax=axis)
        counts = df["machine_failure"].value_counts()
        axis.set_xticks([0, 1], [f"{GROUP_LABELS[i]}\n(n={int(counts.get(i, 0)):,})" for i in (0, 1)])
        axis.set(xlabel="Recorded machine failure status", ylabel="Torque (Nm)", title="Torque distribution by recorded failure status")
        _save_and_close(figure, directory / FIGURE_NAMES[0])
    finally:
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    try:
        # Plot non-failures first so rare failure points remain visible.
        for label in (0, 1):
            group = df.loc[df["machine_failure"] == label]
            axis.scatter(
                group["rotational_speed_rpm"], group["torque_nm"],
                color=GROUP_COLORS[label], alpha=0.25 if label == 0 else 0.8,
                s=14 if label == 0 else 25, edgecolors="none",
                label=f"{GROUP_LABELS[label]} (n={len(group):,})",
            )
        axis.set(xlabel="Rotational speed (RPM)", ylabel="Torque (Nm)", title="Torque and speed by recorded failure status")
        axis.legend()
        _save_and_close(figure, directory / FIGURE_NAMES[1])
    finally:
        plt.close(figure)

    rates = _failure_rates(df)
    figure, axis = plt.subplots(figsize=(7, 5))
    try:
        # Convert within-type failure proportions into percentages.
        bars = axis.bar(rates.index, rates["rate_percent"].fillna(0), color="#3477A5")
        for bar, (_, row) in zip(bars, rates.iterrows()):
            label = f"{row['rate_percent']:.2f}%" if row["count"] else "N/A"
            axis.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center")
        axis.set_xticks(range(3), [f"{kind}\n(n={int(rates.loc[kind, 'count']):,})" for kind in ("L", "M", "H")])
        axis.set_ylim(0, min(110, max(5, float(rates["rate_percent"].fillna(0).max()) * 1.2)))
        axis.set(xlabel="Product type", ylabel="Failure rate (%)", title="Recorded failure percentage within each product type")
        _save_and_close(figure, directory / FIGURE_NAMES[2])
    finally:
        plt.close(figure)

    return [(directory / name).relative_to(REPOSITORY_ROOT.resolve()).as_posix() for name in FIGURE_NAMES]
