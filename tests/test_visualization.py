"""Tests for Amirul's non-overlapping descriptive chart outputs."""

from pathlib import Path

import pandas as pd
from PIL import Image

from industrial_analyzer import visualization
from industrial_analyzer.validation import CANONICAL_COLUMNS


def make_frame() -> pd.DataFrame:
    rows = []
    for uid in range(1, 31):
        failure = int(uid % 5 == 0)
        rows.append([
            uid, f"P{uid:05d}", ("L", "M", "H")[uid % 3],
            298.0 + uid / 20, 308.0 + uid / 20, 1300 + uid,
            65.0 if failure else 35.0, uid, failure, failure, 0, 0, 0, 0,
        ])
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def test_main_charts_are_readable_stable_and_do_not_touch_yahya_files(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization, "REPOSITORY_ROOT", tmp_path)
    frame = make_frame()
    original = frame.copy(deep=True)
    output = tmp_path / "outputs/figures"
    output.mkdir(parents=True)
    yahya_chart = output / "fig_y_01_example.png"
    yahya_chart.write_bytes(b"owned by Yahya")

    paths = visualization.plot_main_insights(frame, "outputs/figures")

    assert len(paths) == 6
    assert all("\\" not in path for path in paths)
    assert [Path(path).name for path in paths] == visualization.FIGURE_NAMES
    assert_frame_equal = pd.testing.assert_frame_equal
    assert_frame_equal(frame, original)
    for relative_path in paths:
        image_path = tmp_path / relative_path
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            image.verify()
    assert yahya_chart.read_bytes() == b"owned by Yahya"
