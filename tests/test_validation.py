import numpy as np

import pandas as pd

from src.validation import check_ranges, plot_missing_before_after, plot_distributions, plot_correlation_heatmap
from src.validation import run_ablation_test


def test_check_ranges_reports_out_of_range_row_indices():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 999.0, 25.0]}, index=[0, 1, 2])

    violations = check_ranges(df, ranges={"Ni_함량_pct": (8, 30)})

    assert violations == {"Ni_함량_pct": [1]}


def test_check_ranges_returns_empty_dict_when_all_within_range():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 25.0]})

    violations = check_ranges(df, ranges={"Ni_함량_pct": (8, 30)})

    assert violations == {}


def test_plot_missing_before_after_creates_png(tmp_path):
    before = pd.DataFrame({"Ni_함량_pct": [1.0, None, 3.0]})
    after = pd.DataFrame({"Ni_함량_pct": [1.0, 2.0, 3.0]})

    output_path = plot_missing_before_after(before, after, tmp_path)

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_distributions_creates_png(tmp_path):
    df = pd.DataFrame({
        "Ni_함량_pct": np.random.default_rng(0).normal(20, 3, 30),
        "황산투입량_L": np.random.default_rng(1).normal(900, 100, 30),
    })

    output_path = plot_distributions(df, ["Ni_함량_pct", "황산투입량_L"], tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_correlation_heatmap_creates_png(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, 30),
        "Co_함량_pct": rng.normal(8, 2, 30),
        "Ni_회수율_pct": rng.normal(85, 5, 30),
    })

    output_path = plot_correlation_heatmap(
        df, ["Ni_함량_pct", "Co_함량_pct"], ["Ni_회수율_pct"], tmp_path
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_ablation_test_returns_r2_and_rmse_per_case_and_target():
    rng = np.random.default_rng(42)
    n = 40
    x1 = rng.normal(20, 3, n)
    x2 = rng.normal(8, 2, n)
    noise = rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "target_1": x1 * 2 + x2 + noise,
    })

    feature_sets = {
        "base": ["feature_a"],
        "base_plus_b": ["feature_a", "feature_b"],
    }

    results = run_ablation_test(df, feature_sets, ["target_1"], n_splits=3, random_state=0)

    assert set(results.keys()) == {("base", "target_1"), ("base_plus_b", "target_1")}
    for metrics in results.values():
        assert "r2_mean" in metrics
        assert "r2_std" in metrics
        assert "rmse_mean" in metrics
        assert isinstance(metrics["r2_mean"], float)
        assert isinstance(metrics["r2_std"], float)
        assert isinstance(metrics["rmse_mean"], float)
        assert metrics["r2_std"] >= 0.0


def test_run_ablation_test_encodes_categorical_features():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "grade": rng.choice(["A", "B", "C"], n),
        "value": rng.normal(10, 2, n),
        "target_1": rng.normal(50, 5, n),
    })

    feature_sets = {"with_grade": ["grade", "value"]}

    results = run_ablation_test(df, feature_sets, ["target_1"], n_splits=3, random_state=0)

    assert ("with_grade", "target_1") in results
