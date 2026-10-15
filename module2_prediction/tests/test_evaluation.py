import numpy as np
import pandas as pd
import pytest

from module2_prediction.evaluation import evaluate_final_model


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


def test_evaluate_final_model_returns_test_metrics():
    df = _make_fixture()
    train_df, val_df, test_df = df.iloc[:30], df.iloc[30:45], df.iloc[45:]

    results = evaluate_final_model(
        train_df, val_df, test_df,
        model_name="RandomForest", model_params={"n_estimators": 50},
    )

    assert set(results.keys()) == {"Ni_회수율_pct", "Co_회수율_pct"}
    for metrics in results.values():
        for key in ["test_r2", "test_rmse", "test_mae"]:
            assert key in metrics


def test_evaluate_final_model_raises_when_target_has_missing_values():
    df = _make_fixture()
    df.loc[0, "Ni_회수율_pct"] = None
    train_df, val_df, test_df = df.iloc[:30], df.iloc[30:45], df.iloc[45:]

    with pytest.raises(ValueError):
        evaluate_final_model(
            train_df, val_df, test_df,
            model_name="RandomForest", model_params={},
        )
