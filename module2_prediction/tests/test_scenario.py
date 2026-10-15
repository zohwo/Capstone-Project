import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from module2_prediction.config import FEATURE_COLUMNS
from module2_prediction.scenario import partial_dependence, predict_blend_scenario


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    grades = np.array(["A"] * (n // 2) + ["C"] * (n - n // 2))
    df = pd.DataFrame({
        "원료등급": grades,
        "Ni_함량_pct": np.where(grades == "A", rng.normal(25, 1, n), rng.normal(12, 1, n)),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": np.where(grades == "A", rng.normal(1, 0.2, n), rng.normal(4, 0.2, n)),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "원료투입량_kg": rng.normal(140, 10, n),
        "황산투입량_L": rng.normal(900, 50, n),
    })
    df["acid_to_ore_ratio"] = df["황산투입량_L"] / df["원료투입량_kg"]
    df["leach_intensity"] = df["침출온도_C"] * df["침출시간_hr"]
    df["total_impurity_pct"] = df["Cu_불순물_pct"] + df["Fe_불순물_pct"] + df["Al_불순물_pct"]
    target_metals = df["Ni_함량_pct"] + df["Co_함량_pct"]
    df["target_purity_ratio"] = target_metals / (target_metals + df["total_impurity_pct"])
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 1, n)
    return df


def test_predict_blend_scenario_returns_float():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    prediction = predict_blend_scenario(model, df, grade_a="A", grade_b="C", ratio_a=0.7)

    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


def test_predict_blend_scenario_rejects_invalid_ratio():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, df, grade_a="A", grade_b="C", ratio_a=1.5)


def test_predict_blend_scenario_rejects_unknown_grade():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, df, grade_a="A", grade_b="Z", ratio_a=0.5)


def test_partial_dependence_varies_predictions_across_values():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    result = partial_dependence(model, base_row, feature="침출온도_C", values=[40, 60, 80])

    assert result.shape == (3, 2)
    assert list(result["침출온도_C"]) == [40, 60, 80]
