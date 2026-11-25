import numpy as np

import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS
import pytest

from sklearn.linear_model import LinearRegression

from module1_preprocessing.validation import check_ranges, check_missing, plot_missing_before_after, plot_distributions, plot_correlation_heatmap
from module1_preprocessing.validation import run_cv_ablation_test
from module1_preprocessing.validation import plot_ablation_results, plot_grade_distribution
from module1_preprocessing.validation import (
    plot_missingno_matrix,
    plot_missingno_bar,
    plot_missingno_heatmap,
    plot_numeric_histograms,
)


def _make_ablation_fixture(n=40, seed=42):
    """engineer_features가 요구하는 전체 컬럼을 갖춘, 결측치가 섞인 실스키마 fixture."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "배치ID": [f"BP-2025-{i:04d}" for i in range(n)],
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


def test_range_display_uses_batch_id_not_row_index():
    from module1_preprocessing.validation import format_range_violations

    df = pd.DataFrame({
        "배치ID": ["BP-0042", "BP-0900"],
        "Ni_함량_pct": [999.0, 20.0],
    }, index=[7, 99])
    result = format_range_violations(df, check_ranges(df))
    assert "BP-0042" in result
    assert "999" in result
    assert "BP-0900" not in result
    assert "변수명" in result and "배치ID" in result and "관측값" in result
    assert format_range_violations(df, {}) == "없음"


def test_check_missing_counts_missing_values_per_column():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, None, 25.0, None],
        "Co_함량_pct": [5.0, 6.0, 7.0, 8.0],
    })

    missing = check_missing(df)

    assert missing == {"Ni_함량_pct": 2}


def test_check_missing_returns_empty_dict_when_nothing_missing():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 25.0]})

    missing = check_missing(df)

    assert missing == {}


def test_check_missing_can_be_scoped_to_specific_columns():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, None],
        "비고": ["무시해야 함", None],
    })

    missing = check_missing(df, columns=["Ni_함량_pct"])

    assert missing == {"Ni_함량_pct": 1}


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


def test_plot_correlation_heatmap_accepts_display_labels(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, 30),
        "Ni_회수율_pct": rng.normal(85, 5, 30),
    })

    output_path = plot_correlation_heatmap(
        df, ["Ni_함량_pct"], ["Ni_회수율_pct"], tmp_path,
        display_labels={"Ni_함량_pct": "니켈 함량"},
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_cv_ablation_test_returns_averaged_metrics_per_case_and_target():
    df = _make_ablation_fixture()

    feature_sets = {
        "base": COMPOSITION_COLUMNS,
        "base_plus_process": COMPOSITION_COLUMNS + PROCESS_COLUMNS,
    }

    results = run_cv_ablation_test(df, feature_sets, ["Ni_회수율_pct"], n_splits=5)

    assert set(results.keys()) == {
        ("base", "Ni_회수율_pct"),
        ("base_plus_process", "Ni_회수율_pct"),
    }
    for metrics in results.values():
        for key in ["cv_r2", "cv_mse", "cv_rmse", "cv_mae"]:
            assert key in metrics
            assert isinstance(metrics[key], float)


def test_run_cv_ablation_test_raises_when_target_has_missing_values():
    df = _make_ablation_fixture()
    df.loc[df.index[0], "Ni_회수율_pct"] = None

    with pytest.raises(ValueError):
        run_cv_ablation_test(df, {"base": COMPOSITION_COLUMNS}, ["Ni_회수율_pct"])


def test_run_cv_ablation_test_is_deterministic_for_fixed_random_state():
    df = _make_ablation_fixture()
    feature_sets = {"base": COMPOSITION_COLUMNS}

    results_1 = run_cv_ablation_test(df, feature_sets, ["Ni_회수율_pct"], random_state=42)
    results_2 = run_cv_ablation_test(df, feature_sets, ["Ni_회수율_pct"], random_state=42)

    assert results_1 == results_2


def test_run_cv_ablation_test_uses_custom_model_factory():
    df = _make_ablation_fixture()
    feature_sets = {"base": COMPOSITION_COLUMNS}

    rf_results = run_cv_ablation_test(df, feature_sets, ["Ni_회수율_pct"])
    lr_results = run_cv_ablation_test(
        df, feature_sets, ["Ni_회수율_pct"],
        model_factory=lambda random_state: LinearRegression(),
    )

    assert rf_results[("base", "Ni_회수율_pct")] != lr_results[("base", "Ni_회수율_pct")]


def test_cv_rejects_test_rows_and_duplicate_batches():
    df = _make_ablation_fixture()
    df['split'] = 'train'
    df.loc[df.index[0], 'split'] = 'test'
    with pytest.raises(ValueError, match='Test'):
        run_cv_ablation_test(df, {'base': COMPOSITION_COLUMNS}, ['Ni_회수율_pct'])
    df['split'] = 'train'
    df['배치ID'] = [f'B{i}' for i in range(len(df))]
    df.loc[1, '배치ID'] = df.loc[0, '배치ID']
    with pytest.raises(ValueError, match='중복'):
        run_cv_ablation_test(df, {'base': COMPOSITION_COLUMNS}, ['Ni_회수율_pct'])


def test_cv_rejects_insufficient_grade_count():
    df = _make_ablation_fixture()
    df['원료등급'] = 'A'
    df.loc[df.index[0], '원료등급'] = 'B'
    with pytest.raises(ValueError, match='표본'):
        run_cv_ablation_test(df, {'base': COMPOSITION_COLUMNS}, ['Ni_회수율_pct'])


def test_cv_reports_fold_spread_and_uses_only_training_imputation(monkeypatch):
    from module1_preprocessing import validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    df = _make_ablation_fixture()
    calls = []
    original = validation.prepare_split

    def record(reference, target):
        calls.append((set(reference.index), set(target.index)))
        return original(reference, target)

    monkeypatch.setattr(validation, 'prepare_split', record)
    result = run_cv_ablation_test(df, {'base': COMPOSITION_COLUMNS}, ['Ni_회수율_pct'],
                                 model_factory=lambda seed: DummyRegressor())
    expected = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (tr, va) in enumerate(cv.split(df, df['원료등급'])):
        train_ids, val_ids = set(df.iloc[tr].index), set(df.iloc[va].index)
        assert calls[2*i] == (train_ids, train_ids)
        assert calls[2*i+1] == (train_ids, val_ids)
        assert train_ids.isdisjoint(val_ids)
        expected.append(np.sqrt(mean_squared_error(df.iloc[va]['Ni_회수율_pct'],
                            np.full(len(va), df.iloc[tr]['Ni_회수율_pct'].mean()))))
    metrics = result[('base', 'Ni_회수율_pct')]
    assert metrics['cv_fold_rmse_mean'] == pytest.approx(np.mean(expected))
    assert metrics['cv_rmse_std'] == pytest.approx(np.std(expected, ddof=1))


def test_plot_ablation_results_creates_png(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    results = {
        ("base_only", "Ni_회수율_pct"): {"cv_mae": 3.8, "cv_r2": -0.5},
        ("base_only", "Co_회수율_pct"): {"cv_mae": 4.3, "cv_r2": -0.2},
        ("base_plus_derived", "Ni_회수율_pct"): {"cv_mae": 2.6, "cv_r2": 0.8},
        ("base_plus_derived", "Co_회수율_pct"): {"cv_mae": 3.1, "cv_r2": 0.7},
    }
    captured = {}
    original = Figure.savefig

    def inspect_plot(fig, *args, **kwargs):
        ax = fig.axes[0]
        captured['heights'] = [bar.get_height() for bar in ax.patches]
        captured['limits'] = ax.get_ylim()
        captured['label'] = ax.get_ylabel()
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect_plot)
    output_path = plot_ablation_results(results, tmp_path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0
    assert captured['heights'] == [3.8, 2.6, 4.3, 3.1]
    assert captured['limits'][0] == 0
    assert captured['limits'][1] > 4.3
    assert captured['label'] == 'CV MAE (%p)'


def test_plot_grade_distribution_creates_png(tmp_path):
    df = pd.DataFrame({"원료등급": ["A", "B", "B", "C", "A", "B"]})

    output_path = plot_grade_distribution(df, tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _make_missingno_fixture(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
    })
    df.loc[[0, 5, 10], "Ni_함량_pct"] = np.nan
    df.loc[[2, 7], "침출온도_C"] = np.nan
    return df


def test_plot_missingno_matrix_creates_png(tmp_path):
    df = _make_missingno_fixture()

    output_path = plot_missingno_matrix(df, tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_missingno_bar_creates_png(tmp_path):
    df = _make_missingno_fixture()

    output_path = plot_missingno_bar(df, tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_missingno_heatmap_creates_png(tmp_path):
    df = _make_missingno_fixture()

    output_path = plot_missingno_heatmap(df, tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_numeric_histograms_creates_png(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, 30),
        "침출온도_C": rng.normal(65, 10, 30),
        "침출시간_hr": rng.normal(4, 1, 30),
    })

    output_path = plot_numeric_histograms(df, ["Ni_함량_pct", "침출온도_C", "침출시간_hr"], tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_numeric_histograms_raises_for_unknown_column(tmp_path):
    df = pd.DataFrame({"Ni_함량_pct": [1.0, 2.0]})

    with pytest.raises(ValueError):
        plot_numeric_histograms(df, ["존재하지않는컬럼"], tmp_path)


def test_composition_sum_flags_review_without_modifying_measurements():
    from module1_preprocessing.validation import check_composition_sum
    frame = pd.DataFrame([
        [20, 10, 5, 15, 5, 5, 5],
        [40, 20, 10, 15, 5, 5, 5],
        [60, 50, 1, 1, 1, 1, 1],
        [20, np.nan, 5, 15, 5, 5, 5],
        [999, 10, 5, 15, 5, 5, 5],
        [np.inf, 10, 5, 15, 5, 5, 5],
        [-1, 10, 5, 15, 5, 5, 5],
    ], columns=COMPOSITION_COLUMNS, index=range(10, 17))
    original = frame.copy(deep=True)
    result = check_composition_sum(frame)
    assert result.loc[10:12, 'composition_sum_review_flag'].tolist() == [False, False, True]
    assert result.loc[13:, 'composition_sum_review_flag'].isna().all()
    assert result.loc[13:, 'composition_sum_pct'].isna().all()
    assert result.loc[12, 'composition_sum_pct'] == 115
    pd.testing.assert_frame_equal(frame, original)
