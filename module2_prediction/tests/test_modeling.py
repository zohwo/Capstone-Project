import numpy as np
import pandas as pd
import pytest

from module2_prediction.modeling import compare_models


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
        "target_purity_ratio": rng.normal(0.8, 0.05, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    return df


def test_compare_models_returns_metrics_for_each_model_and_target():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    results = compare_models(train_df, val_df)

    assert set(results.keys()) == {
        ("RandomForest", "Ni_회수율_pct"), ("RandomForest", "Co_회수율_pct"),
        ("XGBoost", "Ni_회수율_pct"), ("XGBoost", "Co_회수율_pct"),
        ("LightGBM", "Ni_회수율_pct"), ("LightGBM", "Co_회수율_pct"),
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


def test_compare_models_raises_when_target_has_missing_values():
    df = _make_fixture()
    df.loc[0, "Ni_회수율_pct"] = None
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    with pytest.raises(ValueError):
        compare_models(train_df, val_df)
