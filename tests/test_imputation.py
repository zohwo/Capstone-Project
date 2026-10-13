import numpy as np
import pandas as pd
import pytest

from src.imputation import replace_outliers_with_nan


def test_replace_outliers_with_nan_converts_out_of_range_values_to_nan():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, 999.0, 25.0],
        "침출온도_C": [60.0, -5.0, 70.0],
    })

    result = replace_outliers_with_nan(df, ranges={"Ni_함량_pct": (8, 30), "침출온도_C": (40, 95)})

    assert result["Ni_함량_pct"].iloc[0] == 10.0
    assert np.isnan(result["Ni_함량_pct"].iloc[1])
    assert result["Ni_함량_pct"].iloc[2] == 25.0
    assert result["침출온도_C"].iloc[0] == 60.0
    assert np.isnan(result["침출온도_C"].iloc[1])
    assert result["침출온도_C"].iloc[2] == 70.0


def test_replace_outliers_with_nan_keeps_values_within_range_unchanged():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 30.0]})

    result = replace_outliers_with_nan(df, ranges={"Ni_함량_pct": (8, 30)})

    assert result["Ni_함량_pct"].tolist() == [10.0, 20.0, 30.0]


def test_replace_outliers_with_nan_ignores_columns_not_in_dataframe():
    df = pd.DataFrame({"Ni_함량_pct": [10.0]})

    result = replace_outliers_with_nan(df, ranges={"존재하지않는컬럼": (0, 1)})

    assert result["Ni_함량_pct"].tolist() == [10.0]


def _composition_process_columns(n):
    return {
        "Co_함량_pct": [5.0] * n,
        "Li_함량_pct": [3.0] * n,
        "Mn_함량_pct": [9.0] * n,
        "Cu_불순물_pct": [1.0] * n,
        "Fe_불순물_pct": [1.0] * n,
        "Al_불순물_pct": [1.0] * n,
    }


def test_impute_missing_fills_composition_with_grade_group_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A", "C", "C", "C"],
        "Ni_함량_pct": [20.0, None, 24.0, 10.0, None, 12.0],
        **_composition_process_columns(6),
        "침출온도_C": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0],
        "침출시간_hr": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        "원료투입량_kg": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        "황산투입량_L": [700.0, 770.0, 840.0, 910.0, 980.0, 1050.0],
    })

    result = impute_missing_from_reference(df, df)

    # A그룹 Ni 중앙값: median(20.0, 24.0) = 22.0 / C그룹: median(10.0, 12.0) = 11.0
    assert result["Ni_함량_pct"].tolist() == [20.0, 22.0, 24.0, 10.0, 11.0, 12.0]


def test_impute_missing_fills_process_columns_with_overall_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A"],
        "Ni_함량_pct": [20.0] * 3,
        **_composition_process_columns(3),
        "침출온도_C": [60.0, None, 80.0],
        "침출시간_hr": [3.0, 4.0, None],
        "원료투입량_kg": [100.0, 120.0, 140.0],
        "황산투입량_L": [700.0, 840.0, 980.0],
    })

    result = impute_missing_from_reference(df, df)

    assert result["침출온도_C"].tolist() == [60.0, 70.0, 80.0]
    assert result["침출시간_hr"].tolist() == [3.0, 4.0, 3.5]


from src.imputation import impute_missing_from_reference


def _full_row(ni, ore_kg, acid_l, grade="A"):
    return {
        "원료등급": grade,
        "Ni_함량_pct": ni,
        **{k: v[0] for k, v in _composition_process_columns(1).items()},
        "침출온도_C": 60.0,
        "침출시간_hr": 4.0,
        "원료투입량_kg": ore_kg,
        "황산투입량_L": acid_l,
    }


def test_impute_missing_from_reference_predicts_acid_from_reference_regression():
    # reference는 정확히 acid = 7 * ore 관계, target은 결측 1건
    reference = pd.DataFrame([
        _full_row(20.0, 100.0, 700.0),
        _full_row(20.0, 120.0, 840.0),
        _full_row(20.0, 140.0, 980.0),
        _full_row(20.0, 180.0, 1260.0),
    ])
    target = pd.DataFrame([_full_row(20.0, 160.0, None)])

    result = impute_missing_from_reference(reference, target)

    assert result["황산투입량_L"].iloc[0] == pytest.approx(1120.0, rel=1e-6)


def test_impute_missing_from_reference_leaves_complete_column_unchanged():
    reference = pd.DataFrame([
        _full_row(20.0, 100.0, 700.0),
        _full_row(20.0, 120.0, 840.0),
    ])
    target = pd.DataFrame([_full_row(20.0, 100.0, 700.0)])

    result = impute_missing_from_reference(reference, target)

    assert result["황산투입량_L"].tolist() == [700.0]


def test_impute_missing_from_reference_uses_reference_stats_not_target_values():
    # reference의 Ni 중앙값은 22.0 (median(20, 24))
    reference = pd.DataFrame([
        _full_row(20.0, 100.0, 700.0),
        _full_row(24.0, 100.0, 700.0),
    ])
    # target의 다른 행에 극단값(9999.0)이 있어도, 결측 행 대치에는 영향을 주면 안 된다
    target = pd.DataFrame([
        _full_row(None, 100.0, 700.0),
        _full_row(9999.0, 100.0, 700.0),
    ])

    result = impute_missing_from_reference(reference, target)

    assert result["Ni_함량_pct"].iloc[0] == pytest.approx(22.0)


from src.imputation import interpolate_analysis_date


def test_interpolate_analysis_date_fills_gap_using_batch_order():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003", "BP-2025-0004"],
        "분석일자": [
            pd.Timestamp("2025-01-01"),
            pd.NaT,
            pd.NaT,
            pd.Timestamp("2025-01-04"),
        ],
    })

    result = interpolate_analysis_date(df)

    assert result["분석일자"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    ]


def test_interpolate_analysis_date_extrapolates_boundary_gap():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003"],
        "분석일자": [pd.NaT, pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
    })

    result = interpolate_analysis_date(df)

    assert result["분석일자"].isna().sum() == 0
