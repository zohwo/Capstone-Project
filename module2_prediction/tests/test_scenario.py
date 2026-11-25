import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from module2_prediction.config import FEATURE_COLUMNS
from module2_prediction.scenario import partial_dependence, predict_blend_scenario, predict_process_scenario, build_blend_features


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
    df["feed_impurity_g_per_l"] = 10 * df["total_impurity_pct"] / df["acid_to_ore_ratio"]
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 1, n)
    return df


def test_predict_blend_scenario_returns_float():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    prediction = predict_blend_scenario(model, df.iloc[0], df.iloc[-1], total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4, ratio_a=0.7)

    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


def test_predict_blend_scenario_rejects_invalid_ratio():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, df.iloc[0], df.iloc[-1], total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4, ratio_a=1.5)


def test_predict_blend_scenario_rejects_missing_assay():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, {}, df.iloc[-1], ratio_a=0.5, total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4)


def test_partial_dependence_varies_predictions_across_values():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    result = partial_dependence(model, base_row, feature="침출온도_C", values=[40, 60, 80])

    assert result.shape == (3, 2)
    assert list(result["침출온도_C"]) == [40, 60, 80]


def test_partial_dependence_accepts_generator_values():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    result = partial_dependence(model, base_row, feature="침출온도_C", values=(v for v in [40, 60, 80]))

    assert result.shape == (3, 2)
    assert list(result["침출온도_C"]) == [40, 60, 80]


class _EchoLeachIntensityModel:
    """leach_intensity 컬럼을 그대로 반환해, 호출자가 넘긴 X에 어떤 값이 들어갔는지 검증한다."""

    def predict(self, X):
        return X["leach_intensity"].to_numpy()


def test_acid_scenario_recomputes_feed_impurity():
    class Echo:
        def predict(self, X):
            return X['feed_impurity_g_per_l'].to_numpy()
    row = _make_fixture().iloc[0].copy()
    result = predict_process_scenario(Echo(), row, '황산투입량_L', [500, 1000])
    assert result.prediction.iloc[0] == pytest.approx(2 * result.prediction.iloc[1])


def test_predict_process_scenario_recomputes_leach_intensity_for_temperature():
    df = _make_fixture()
    base_row = df.iloc[0].copy()
    base_row["침출시간_hr"] = 4.0

    result = predict_process_scenario(
        _EchoLeachIntensityModel(), base_row, feature="침출온도_C", values=[40, 80]
    )

    np.testing.assert_allclose(result["prediction"].to_numpy(), [160.0, 320.0])


def test_predict_process_scenario_recomputes_leach_intensity_for_time():
    df = _make_fixture()
    base_row = df.iloc[0].copy()
    base_row["침출온도_C"] = 60.0

    result = predict_process_scenario(
        _EchoLeachIntensityModel(), base_row, feature="침출시간_hr", values=[2, 8]
    )

    np.testing.assert_allclose(result["prediction"].to_numpy(), [120.0, 480.0])


def test_predict_process_scenario_rejects_out_of_range_swept_value():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    with pytest.raises(ValueError, match="침출시간_hr"):
        predict_process_scenario(model, base_row, feature="침출시간_hr", values=[-10, 2])


def test_predict_process_scenario_rejects_out_of_range_base_row():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0].copy()
    base_row["침출시간_hr"] = -10

    with pytest.raises(ValueError, match="침출시간_hr"):
        predict_process_scenario(model, base_row, feature="침출온도_C", values=[50, 60])


def test_predict_process_scenario_rejects_non_process_feature():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    with pytest.raises(ValueError):
        predict_process_scenario(model, base_row, feature="Ni_함량_pct", values=[10, 20])


def test_blend_uses_actual_assays_and_keeps_batch_conditions_fixed():
    a, c = _make_fixture().iloc[[0, -1]].copy().iloc[0], _make_fixture().iloc[-1].copy()
    a["Ni_함량_pct"], c["Ni_함량_pct"] = 20., 10.
    a["침출온도_C"], c["침출온도_C"] = 90., 40.
    conditions = dict(total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4)
    first = build_blend_features(a, c, .7, **conditions).iloc[0]
    second = build_blend_features(a, c, .3, **conditions).iloc[0]
    assert first["Ni_함량_pct"] == pytest.approx(17.)
    assert second["Ni_함량_pct"] == pytest.approx(13.)
    for col, expected in {"침출온도_C":70, "침출시간_hr":4, "원료투입량_kg":100,
                          "황산투입량_L":600, "leach_intensity":280, "acid_to_ore_ratio":6}.items():
        assert first[col] == second[col] == expected
    assert first["feed_impurity_g_per_l"] == pytest.approx(10 * first["total_impurity_pct"] / 6)


@pytest.mark.parametrize("change", [{"total_ore_kg":0}, {"acid_l":float("nan")}, {"temperature_c":101}, {"time_hr":-1}])
def test_blend_rejects_invalid_conditions(change):
    df = _make_fixture()
    conditions = dict(total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4)
    conditions.update(change)
    with pytest.raises(ValueError):
        build_blend_features(df.iloc[0], df.iloc[-1], .7, **conditions)


@pytest.mark.parametrize("value", [float("nan"), -1, 999, 100])
def test_blend_rejects_invalid_composition(value):
    df = _make_fixture()
    a = df.iloc[0].copy()
    a["Ni_함량_pct"] = value
    with pytest.raises(ValueError):
        build_blend_features(a, df.iloc[-1], .7, total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4)
