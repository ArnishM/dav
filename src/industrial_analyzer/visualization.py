"""Create Amirul's six descriptive figures with the ``fig_`` prefix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .validation import MEASUREMENT_COLUMNS, validate_ai4i_data


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIGURE_NAMES = [
    "fig_01_failure_rate_by_type.png",
    "fig_02_sensor_distributions_by_failure.png",
    "fig_03_torque_vs_rotational_speed.png",
    "fig_04_air_vs_process_temperature.png",
    "fig_05_tool_wear_by_failure.png",
    "fig_06_sensor_correlation.png",
]


def _output_directory(output_dir: str | Path) -> Path:
    if not str(output_dir).strip():
        raise ValueError("output_dir must be a nonempty path")
    directory = Path(output_dir).expanduser()
    if not directory.is_absolute():
        directory = REPOSITORY_ROOT / directory
    directory = directory.resolve()
    if not directory.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("output_dir must be inside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save(fig: plt.Figure, directory: Path, filename: str) -> str:
    path = directory / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def plot_main_insights(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
) -> list[str]:
    """Write six descriptive figures and return repository-relative paths.

    The figures describe observed associations. They do not establish causes,
    forecasts, or equipment safety limits. Sensor extremes remain in the data.
    """
    validate_ai4i_data(df)
    directory = _output_directory(output_dir)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[str] = []

    counts = df.groupby("type", observed=False)["machine_failure"].agg(
        records="size", failures="sum"
    ).reindex(["L", "M", "H"], fill_value=0)
    rates = counts["failures"].div(counts["records"].replace(0, pd.NA)).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(rates.index, rates.values, color="#4472C4")
    ax.set_ylim(0, max(1.0, float(rates.max()) * 1.22))
    ax.set(title="Recorded machine failure rate by product type",
           xlabel="Product type", ylabel="Recorded failures (%)")
    for bar, (_, row), rate in zip(bars, counts.iterrows(), rates.values):
        ax.annotate(f"{rate:.2f}%\nn={int(row['records'])}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    paths.append(_save(fig, directory, FIGURE_NAMES[0]))

    sensor_names = [
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm",
    ]
    labels = {
        "air_temperature_k": "Air temperature (K)",
        "process_temperature_k": "Process temperature (K)",
        "rotational_speed_rpm": "Rotational speed (rpm)",
        "torque_nm": "Torque (Nm)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_data = df.assign(machine_failure_label=df["machine_failure"].map(
        {0: "No failure", 1: "Failure"}
    ))
    for ax, column in zip(axes.flat, sensor_names):
        sns.boxplot(data=plot_data, x="machine_failure_label", y=column,
                    order=["No failure", "Failure"],
                    showfliers=True, ax=ax, color="#8FB9A8")
        ax.set(xlabel="Recorded machine failure", ylabel=labels[column], title=labels[column])
    fig.suptitle("Sensor distributions by recorded failure label")
    fig.tight_layout()
    paths.append(_save(fig, directory, FIGURE_NAMES[1]))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="rotational_speed_rpm", y="torque_nm",
                    hue="machine_failure", hue_order=[0, 1], alpha=0.55,
                    palette={0: "#4472C4", 1: "#C0504D"}, ax=ax)
    ax.set(title="Torque and rotational speed by recorded failure label",
           xlabel="Rotational speed (rpm)", ylabel="Torque (Nm)")
    ax.legend(title="Machine failure\n(0=no, 1=yes)")
    paths.append(_save(fig, directory, FIGURE_NAMES[2]))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="air_temperature_k", y="process_temperature_k",
                    hue="machine_failure", hue_order=[0, 1], alpha=0.55,
                    palette={0: "#4472C4", 1: "#C0504D"}, ax=ax)
    ax.set(title="Air and process temperatures by recorded failure label",
           xlabel="Air temperature (K)", ylabel="Process temperature (K)")
    ax.legend(title="Machine failure\n(0=no, 1=yes)")
    paths.append(_save(fig, directory, FIGURE_NAMES[3]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=plot_data, x="machine_failure_label", y="tool_wear_min",
                order=["No failure", "Failure"],
                showfliers=True, ax=ax, color="#E6B566")
    ax.set(title="Tool wear by recorded failure label",
           xlabel="Recorded machine failure", ylabel="Tool wear (min)")
    paths.append(_save(fig, directory, FIGURE_NAMES[4]))

    correlation_columns = [*MEASUREMENT_COLUMNS, "machine_failure"]
    correlation = df.loc[:, correlation_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set(title="Sensor and recorded failure-label correlations")
    paths.append(_save(fig, directory, FIGURE_NAMES[5]))

    return paths
