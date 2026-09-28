import pandas as pd
import pytest

from src.features import engineer_features


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
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert result["acid_to_ore_ratio"].iloc[0] == pytest.approx(7.0)
    assert result["leach_intensity"].iloc[0] == pytest.approx(240.0)
    assert result["total_impurity_pct"].iloc[0] == pytest.approx(3.0)
    assert result["target_purity_ratio"].iloc[0] == pytest.approx(25.0 / 28.0)
    assert result["analysis_month"].iloc[0] == 3


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
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    engineer_features(df)

    assert "acid_to_ore_ratio" not in df.columns
