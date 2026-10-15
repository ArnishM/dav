"""Artificial fixtures and isolated chart outputs, never project results."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pandas.testing import assert_frame_equal

from industrial_analyzer import advanced_analysis as analysis
from industrial_analyzer.features import CANONICAL_COLUMNS, engineer_features


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "REPOSITORY_ROOT", tmp_path)


@pytest.fixture
def artificial_data():
    rows = []
    for i, kind in enumerate(["L"] * 2 + ["M"] * 4 + ["H"] * 6):
        failure = int(i in (0, 2))
        rows.append([i + 1, f"ARTIFICIAL-{i}", kind, 300.0, 310.0, 1500 + i, 40.0 + i, i, failure, 0, 0, 0, 0, 0])
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS, index=pd.Index(range(20, 32), name="artificial_index"))


def test_signature():
    assert str(inspect.signature(analysis.plot_failure_insights)) == "(df: pandas.DataFrame, output_dir: str) -> list[str]"


@pytest.mark.parametrize("derived", [False, True])
def test_files_readable_preservation_and_repeated_runs(artificial_data, derived):
    data = engineer_features(artificial_data) if derived else artificial_data
    original = data.copy(deep=True)
    root = analysis.REPOSITORY_ROOT
    directory = root / "outputs/figures"
    directory.mkdir(parents=True)
    teammate = directory / "fig_01_amirul.png"
    teammate.write_bytes(b"teammate sentinel")
    open_figures = plt.get_fignums()
    paths = analysis.plot_failure_insights(data, "outputs/figures")
    assert paths == [f"outputs/figures/{name}" for name in analysis.FIGURE_NAMES]
    assert all("\\" not in path and not path.startswith("/") for path in paths)
    for path in paths:
        file = root / path
        assert file.stat().st_size > 0
        with Image.open(file) as image:
            assert image.format == "PNG"
            image.verify()
        with Image.open(file) as image:
            image.load()
            assert image.width > 100 and image.height > 100
    assert analysis.plot_failure_insights(data, "outputs/figures") == paths
    assert len(list(directory.glob("fig_y_*.png"))) == 3
    assert teammate.read_bytes() == b"teammate sentinel"
    assert_frame_equal(data, original)
    assert plt.get_fignums() == open_figures


@pytest.mark.parametrize("column", analysis.DERIVED_COLUMNS)
def test_partial_derived(artificial_data, column):
    artificial_data[column] = 0
    with pytest.raises(ValueError, match="Both derived"):
        analysis.plot_failure_insights(artificial_data, "charts")


@pytest.mark.parametrize("column", analysis.DERIVED_COLUMNS)
@pytest.mark.parametrize("value", [999.0, np.inf, np.nan, "10", True])
def test_invalid_derived(artificial_data, column, value):
    data = engineer_features(artificial_data).astype({column: object})
    data.loc[20, column] = value
    with pytest.raises(ValueError, match=column):
        analysis.plot_failure_insights(data, "charts")


def test_missing_extra_and_reordered_columns(artificial_data):
    for data in (
        artificial_data.drop(columns="uid"), artificial_data.assign(unrelated=1),
        artificial_data[list(reversed(CANONICAL_COLUMNS))],
    ):
        with pytest.raises(ValueError, match="canonical columns"):
            analysis.plot_failure_insights(data, "charts")


@pytest.mark.parametrize("column,value", [
    ("torque_nm", np.nan), ("torque_nm", np.inf), ("torque_nm", -1),
    ("torque_nm", "40"), ("machine_failure", 2), ("type", "X"),
    ("uid", 2), ("product_id", ""),
])
def test_invalid_canonical_values(artificial_data, column, value):
    data = artificial_data.astype({column: object})
    data.loc[20, column] = value
    with pytest.raises(ValueError, match=column):
        analysis.plot_failure_insights(data, "charts")


def test_empty_wrong_type_and_duplicate_columns(artificial_data):
    for data in (None, artificial_data.iloc[:0], pd.concat([artificial_data, artificial_data[["uid"]]], axis=1)):
        with pytest.raises(ValueError):
            analysis.plot_failure_insights(data, "charts")


@pytest.mark.parametrize("path", ["", " ", None, "../escape", "."])
def test_invalid_paths(artificial_data, path):
    with pytest.raises(ValueError):
        analysis.plot_failure_insights(artificial_data, path)


def test_absolute_path_and_directory_creation(artificial_data, monkeypatch):
    root = analysis.REPOSITORY_ROOT
    nested = root / "notebooks"
    nested.mkdir()
    monkeypatch.chdir(nested)
    paths = analysis.plot_failure_insights(artificial_data, str(root / "new/charts"))
    assert paths == [f"new/charts/{name}" for name in analysis.FIGURE_NAMES]
    with pytest.raises(ValueError, match="inside the repository"):
        analysis.plot_failure_insights(artificial_data, str(root.parent / "escape"))


@pytest.mark.parametrize("mode", ["zero_failures", "single_category", "all_failures"])
def test_sparse_groups(artificial_data, mode):
    data = artificial_data.copy()
    if mode == "zero_failures":
        data["machine_failure"] = 0
    elif mode == "all_failures":
        data["machine_failure"] = 1
    else:
        data["type"] = "L"
    paths = analysis.plot_failure_insights(data, "charts")
    assert len(paths) == 3
    rates = analysis._failure_rates(data)
    if mode == "zero_failures":
        assert rates.rate_percent.tolist() == [0.0, 0.0, 0.0]
    elif mode == "single_category":
        assert rates.loc["M", "count"] == 0 and np.isnan(rates.loc["M", "rate_percent"])
    else:
        assert rates.rate_percent.tolist() == [100.0] * 3


def test_failure_rate_denominators_and_plotted_values(artificial_data, monkeypatch):
    rates = analysis._failure_rates(artificial_data)
    assert rates.index.tolist() == ["L", "M", "H"]
    assert rates["count"].tolist() == [2, 4, 6]
    assert rates["sum"].tolist() == [1, 1, 0]
    assert rates.rate_percent.tolist() == [50.0, 25.0, 0.0]
    from matplotlib.axes import Axes
    original_bar = Axes.bar
    heights = []
    def record_bar(self, x, height, *args, **kwargs):
        heights.extend(list(height))
        return original_bar(self, x, height, *args, **kwargs)
    monkeypatch.setattr(Axes, "bar", record_bar)
    analysis.plot_failure_insights(artificial_data, "charts")
    assert heights == [50.0, 25.0, 0.0]


def test_scatter_order_and_labels(artificial_data, monkeypatch):
    from matplotlib.axes import Axes
    original_scatter = Axes.scatter
    groups = []
    def record_scatter(self, x, y, *args, **kwargs):
        if "label" in kwargs:
            groups.append((kwargs["label"], list(x)))
        return original_scatter(self, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    analysis.plot_failure_insights(artificial_data, "charts")
    assert groups[0][0] == "No failure (n=10)"
    assert groups[1][0] == "Failure (n=2)"
    assert groups[1][1] == artificial_data.loc[artificial_data.machine_failure == 1, "rotational_speed_rpm"].tolist()


def test_save_failure_closes_figures(artificial_data, monkeypatch):
    from matplotlib.figure import Figure
    def failed_save(*args, **kwargs):
        raise PermissionError("artificial output denial")
    monkeypatch.setattr(Figure, "savefig", failed_save)
    before = plt.get_fignums()
    with pytest.raises(OSError, match="Could not save chart"):
        analysis.plot_failure_insights(artificial_data, "charts")
    assert plt.get_fignums() == before


def test_output_is_file(artificial_data):
    (analysis.REPOSITORY_ROOT / "file").write_text("preserve")
    with pytest.raises(OSError):
        analysis.plot_failure_insights(artificial_data, "file")


def test_chart_destination_is_directory(artificial_data):
    directory = analysis.REPOSITORY_ROOT / "charts"
    (directory / analysis.FIGURE_NAMES[0]).mkdir(parents=True)
    with pytest.raises(ValueError, match="regular file"):
        analysis.plot_failure_insights(artificial_data, "charts")
