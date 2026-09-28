import numpy as np
import pandas as pd

from src.imputation import flag_outliers


def test_flag_outliers_converts_out_of_range_values_to_nan():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, 999.0, 25.0],
        "침출온도_C": [60.0, -5.0, 70.0],
    })

    result = flag_outliers(df, ranges={"Ni_함량_pct": (8, 30), "침출온도_C": (40, 95)})

    assert result["Ni_함량_pct"].iloc[0] == 10.0
    assert np.isnan(result["Ni_함량_pct"].iloc[1])
    assert result["Ni_함량_pct"].iloc[2] == 25.0
    assert result["침출온도_C"].iloc[0] == 60.0
    assert np.isnan(result["침출온도_C"].iloc[1])
    assert result["침출온도_C"].iloc[2] == 70.0


def test_flag_outliers_keeps_values_within_range_unchanged():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 30.0]})

    result = flag_outliers(df, ranges={"Ni_함량_pct": (8, 30)})

    assert result["Ni_함량_pct"].tolist() == [10.0, 20.0, 30.0]


def test_flag_outliers_ignores_columns_not_in_dataframe():
    df = pd.DataFrame({"Ni_함량_pct": [10.0]})

    result = flag_outliers(df, ranges={"존재하지않는컬럼": (0, 1)})

    assert result["Ni_함량_pct"].tolist() == [10.0]


from src.imputation import impute_missing


def test_impute_missing_fills_composition_with_grade_group_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A", "C", "C", "C"],
        "Ni_함량_pct": [20.0, None, 24.0, 10.0, None, 12.0],
        "Co_함량_pct": [5.0] * 6,
        "Li_함량_pct": [3.0] * 6,
        "Mn_함량_pct": [9.0] * 6,
        "Cu_불순물_pct": [1.0] * 6,
        "Fe_불순물_pct": [1.0] * 6,
        "Al_불순물_pct": [1.0] * 6,
        "침출온도_C": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0],
        "침출시간_hr": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    })

    result = impute_missing(df)

    # A그룹 Ni 중앙값: median(20.0, 24.0) = 22.0 / C그룹: median(10.0, 12.0) = 11.0
    assert result["Ni_함량_pct"].tolist() == [20.0, 22.0, 24.0, 10.0, 11.0, 12.0]


def test_impute_missing_fills_process_columns_with_overall_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A"],
        "Ni_함량_pct": [20.0] * 3,
        "Co_함량_pct": [5.0] * 3,
        "Li_함량_pct": [3.0] * 3,
        "Mn_함량_pct": [9.0] * 3,
        "Cu_불순물_pct": [1.0] * 3,
        "Fe_불순물_pct": [1.0] * 3,
        "Al_불순물_pct": [1.0] * 3,
        "침출온도_C": [60.0, None, 80.0],
        "침출시간_hr": [3.0, 4.0, None],
    })

    result = impute_missing(df)

    assert result["침출온도_C"].tolist() == [60.0, 70.0, 80.0]
    assert result["침출시간_hr"].tolist() == [3.0, 4.0, 3.5]
