import numpy as np
import pytest

import pandas as pd

from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS
from src.validation import check_ranges, plot_missing_before_after, plot_distributions, plot_correlation_heatmap
from src.validation import run_ablation_test, _fold_safe_prepare
from src.validation import plot_ablation_results, plot_grade_distribution


def _make_ablation_fixture(n=40, seed=42):
    """engineer_features가 요구하는 전체 컬럼을 갖춘, 결측치가 섞인 실스키마 fixture."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "원료등급": rng.choice(["A", "B", "C"], n),
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "원료투입량_kg": rng.normal(140, 30, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "분석일자": pd.date_range("2025-01-01", periods=n, freq="D"),
    })
    df["황산투입량_L"] = df["원료투입량_kg"] * 7 + rng.normal(0, 10, n)
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)

    df.loc[[0, 5, 10], "Ni_함량_pct"] = np.nan
    df.loc[[2, 7], "침출온도_C"] = np.nan
    df.loc[3, "황산투입량_L"] = np.nan

    return df


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
    df = _make_ablation_fixture()

    feature_sets = {
        "base": COMPOSITION_COLUMNS,
        "base_plus_process": COMPOSITION_COLUMNS + PROCESS_COLUMNS,
    }

    results = run_ablation_test(df, feature_sets, ["Ni_회수율_pct"], n_splits=3, random_state=0)

    assert set(results.keys()) == {
        ("base", "Ni_회수율_pct"),
        ("base_plus_process", "Ni_회수율_pct"),
    }
    for metrics in results.values():
        assert "r2_mean" in metrics
        assert "r2_std" in metrics
        assert "rmse_mean" in metrics
        assert isinstance(metrics["r2_mean"], float)
        assert isinstance(metrics["r2_std"], float)
        assert isinstance(metrics["rmse_mean"], float)
        assert metrics["r2_std"] >= 0.0


def test_run_ablation_test_encodes_categorical_features():
    df = _make_ablation_fixture()

    feature_sets = {"with_grade": COMPOSITION_COLUMNS + ["원료등급"]}

    results = run_ablation_test(df, feature_sets, ["Ni_회수율_pct"], n_splits=3, random_state=0)

    assert ("with_grade", "Ni_회수율_pct") in results


def test_fold_safe_prepare_imputes_missing_value_from_train_group_median_only():
    df = _make_ablation_fixture()
    train_df = df.iloc[:30]
    apply_df = df.iloc[30:].copy()

    target_row = apply_df.index[0]
    grade = apply_df.loc[target_row, "원료등급"]
    apply_df.loc[target_row, "Ni_함량_pct"] = np.nan

    result = _fold_safe_prepare(train_df, apply_df)

    expected = train_df.loc[train_df["원료등급"] == grade, "Ni_함량_pct"].median()
    assert result.loc[target_row, "Ni_함량_pct"] == pytest.approx(expected)


def test_fold_safe_prepare_ignores_apply_df_values_when_imputing():
    df = _make_ablation_fixture()
    train_df = df.iloc[:30]

    # apply_df 쪽 값을 극단적으로 바꿔도 train_df 통계만 쓰이므로 대치값이 그대로여야 한다
    apply_a = df.iloc[30:].copy()
    apply_b = df.iloc[30:].copy()
    apply_b["Ni_함량_pct"] = 9999.0

    target_row = apply_a.index[0]
    apply_a.loc[target_row, "Ni_함량_pct"] = np.nan
    apply_b.loc[target_row, "Ni_함량_pct"] = np.nan

    result_a = _fold_safe_prepare(train_df, apply_a)
    result_b = _fold_safe_prepare(train_df, apply_b)

    assert result_a.loc[target_row, "Ni_함량_pct"] == result_b.loc[target_row, "Ni_함량_pct"]


def test_fold_safe_prepare_returns_engineered_columns():
    df = _make_ablation_fixture()
    train_df = df.iloc[:30]
    apply_df = df.iloc[30:]

    result = _fold_safe_prepare(train_df, apply_df)

    for col in ["acid_to_ore_ratio", "leach_intensity", "total_impurity_pct", "target_purity_ratio"]:
        assert col in result.columns
        assert result[col].isna().sum() == 0


def test_plot_ablation_results_creates_png(tmp_path):
    results = {
        ("base_only", "Ni_회수율_pct"): {"r2_mean": 0.58, "r2_std": 0.04, "rmse_mean": 3.8},
        ("base_only", "Co_회수율_pct"): {"r2_mean": 0.60, "r2_std": 0.04, "rmse_mean": 4.3},
        ("base_plus_derived", "Ni_회수율_pct"): {"r2_mean": 0.81, "r2_std": 0.03, "rmse_mean": 2.6},
        ("base_plus_derived", "Co_회수율_pct"): {"r2_mean": 0.79, "r2_std": 0.02, "rmse_mean": 3.1},
    }

    output_path = plot_ablation_results(results, tmp_path)

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_grade_distribution_creates_png(tmp_path):
    df = pd.DataFrame({"원료등급": ["A", "B", "B", "C", "A", "B"]})

    output_path = plot_grade_distribution(df, tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
