import numpy as np
import pandas as pd
import pytest

from module2_prediction.config import TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES, compare_models


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "acid_to_ore_ratio": rng.normal(7, 0.5, n),
        "leach_intensity": rng.normal(260, 40, n),
        "total_impurity_pct": rng.normal(6, 1, n),
        "feed_impurity_g_per_l": rng.normal(10.0, 3.0, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["원료등급"] = ["A", "B", "C"] * (n // 3)
    df["원료투입량_kg"] = 140.0
    df["황산투입량_L"] = 980.0
    df["분석일자"] = pd.Timestamp("2025-01-01")
    return df


def test_compare_models_returns_metrics_for_each_model_and_target():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    results = compare_models(train_df, val_df)

    assert set(results.keys()) == {
        (model_name, target)
        for model_name in MODEL_FACTORIES
        for target in TARGET_COLUMNS
    }
    for metrics in results.values():
        for key in ["val_r2", "val_rmse", "val_mae"]:
            assert key in metrics
            assert isinstance(metrics[key], float)


def test_compare_models_applies_custom_hyperparameters():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    results = compare_models(
        train_df, val_df,
        model_params={"RandomForest": {"n_estimators": 10}},
    )

    assert ("RandomForest", "Ni_회수율_pct") in results


def test_lightgbm_factory_uses_gain_importance_type():
    model = MODEL_FACTORIES["LightGBM"]()

    assert model.importance_type == "gain"


def test_compare_models_raises_when_target_has_missing_values():
    df = _make_fixture()
    df.loc[0, "Ni_회수율_pct"] = None
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    with pytest.raises(ValueError):
        compare_models(train_df, val_df)


def test_compare_models_raises_with_single_validation_row():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:41]

    with pytest.raises(ValueError, match="Validation"):
        compare_models(train_df, val_df)


def test_compare_models_reports_nonfinite_r2_for_constant_validation_target():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40].copy(), df.iloc[40:].copy()
    val_df["Ni_회수율_pct"] = 70.0

    results = compare_models(train_df, val_df, target_columns=["Ni_회수율_pct"])

    for metrics in results.values():
        assert np.isneginf(metrics["val_r2"])
        assert np.isfinite(metrics["val_rmse"])


def test_compare_models_reuses_train_based_preprocessing(monkeypatch):
    from module2_prediction import modeling

    df = _make_fixture()
    df.loc[[0, 45], 'Ni_함량_pct'] = np.nan
    train, val = df.iloc[:40].copy(), df.iloc[40:].copy()
    train_before, val_before = train.copy(), val.copy()
    original = modeling.prepare_split
    calls = []

    def record(reference, target):
        calls.append((reference, target))
        return original(reference, target)

    monkeypatch.setattr(modeling, 'prepare_split', record)
    result = modeling.compare_models(train, val)

    assert len(calls) == 2
    assert calls[0][0] is train and calls[0][1] is train
    assert calls[1][0] is train and calls[1][1] is val
    assert len(result) == len(MODEL_FACTORIES) * len(TARGET_COLUMNS)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(val, val_before)
