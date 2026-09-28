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
