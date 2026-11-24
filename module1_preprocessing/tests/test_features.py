import numpy as np
import pandas as pd
import pytest

from module1_preprocessing.features import engineer_features


def test_engineer_features_computes_all_derived_columns():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "Li_함량_pct": [4.0],
        "Mn_함량_pct": [10.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert result["acid_to_ore_ratio"].iloc[0] == pytest.approx(7.0)
    assert result["leach_intensity"].iloc[0] == pytest.approx(240.0)
    assert result["total_impurity_pct"].iloc[0] == pytest.approx(3.0)
    assert result["target_purity_ratio"].iloc[0] == pytest.approx(25.0 / 28.0)
    assert result["analysis_month"].iloc[0] == 3
    assert result["acid_per_target_L_kg"].iloc[0] == pytest.approx(28.0)
    assert result["acid_time"].iloc[0] == pytest.approx(28.0)
    assert result["feed_impurity_g_per_l"].iloc[0] == pytest.approx(30 / 7)
    assert result["Fe_Ni_ratio"].iloc[0] == pytest.approx(0.05)
    assert result["Cu_Ni_ratio"].iloc[0] == pytest.approx(0.05)
    assert result["Al_Ni_ratio"].iloc[0] == pytest.approx(0.05)


def test_engineer_features_computes_distinct_impurity_ratios():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [2.0],
        "Fe_불순물_pct": [4.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "Li_함량_pct": [4.0],
        "Mn_함량_pct": [10.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert result["Cu_Ni_ratio"].iloc[0] == pytest.approx(2.0 / 20.0)
    assert result["Fe_Ni_ratio"].iloc[0] == pytest.approx(4.0 / 20.0)
    assert result["Al_Ni_ratio"].iloc[0] == pytest.approx(1.0 / 20.0)


def test_engineer_features_zero_denominator_returns_nan_not_inf():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [0.0],  # Ni 분모가 0 -> Fe/Cu/Al_Ni_ratio는 NaN이어야 함(inf 아님)
        "Co_함량_pct": [5.0],
        "Li_함량_pct": [4.0],
        "Mn_함량_pct": [10.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert np.isnan(result["Fe_Ni_ratio"].iloc[0])
    assert np.isnan(result["Cu_Ni_ratio"].iloc[0])
    assert np.isnan(result["Al_Ni_ratio"].iloc[0])
    assert not np.isinf(result["Fe_Ni_ratio"].iloc[0])


def test_engineer_features_negative_ore_mass_returns_nan_not_negative_ratio():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [-10.0],  # 음수 분모 -> acid_to_ore_ratio는 NaN이어야 함
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "Li_함량_pct": [4.0],
        "Mn_함량_pct": [10.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert np.isnan(result["acid_to_ore_ratio"].iloc[0])
    assert np.isnan(result["feed_impurity_g_per_l"].iloc[0])


def test_engineer_features_does_not_mutate_input():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "Li_함량_pct": [4.0],
        "Mn_함량_pct": [10.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    engineer_features(df)

    assert "acid_to_ore_ratio" not in df.columns


def test_new_acid_features_reject_zero_target_mass():
    frame = pd.DataFrame({
        "황산투입량_L": [700., 700.], "원료투입량_kg": [100., 0.],
        "침출온도_C": [60., 60.], "침출시간_hr": [4., 4.],
        "Cu_불순물_pct": [1., 1.], "Fe_불순물_pct": [1., 1.],
        "Al_불순물_pct": [1., 1.], "Ni_함량_pct": [0., 20.],
        "Co_함량_pct": [0., 5.], "분석일자": pd.to_datetime(["2025-01-01"] * 2),
    })
    result = engineer_features(frame)
    assert result["acid_per_target_L_kg"].isna().all()
    assert np.isnan(result["acid_time"].iloc[1])
    assert np.isnan(result["feed_impurity_g_per_l"].iloc[1])
