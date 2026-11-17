import numpy as np
import pandas as pd
import pytest
from module1_preprocessing.imputation import replace_outliers_with_nan
from module1_preprocessing.imputation import impute_missing_from_reference
from module1_preprocessing.imputation import interpolate_analysis_date
from module1_preprocessing.config import (
    NORMAL_RANGES, REFERENCE_EXCEPTION_RANGES, BASE_FEATURES,
    DERIVED_FEATURES, TARGET_COLUMNS,
)
from module1_preprocessing.validation import check_ranges


def test_replace_outliers_with_nan_converts_unapproved_values_to_nan():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, 999.0, 25.0],
        "침출온도_C": [60.0, -5.0, 70.0],
    })

    result = replace_outliers_with_nan(
        df,
        ranges={"Ni_함량_pct": (8, 30), "침출온도_C": (40, 95)},
        reference_exception_ranges={"Ni_함량_pct": ((30, 38.2),)},
    )

    assert result["Ni_함량_pct"].iloc[0] == 10.0
    assert np.isnan(result["Ni_함량_pct"].iloc[1])
    assert result["Ni_함량_pct"].iloc[2] == 25.0
    assert result["침출온도_C"].iloc[0] == 60.0
    assert np.isnan(result["침출온도_C"].iloc[1])
    assert result["침출온도_C"].iloc[2] == 70.0


def test_replace_outliers_with_nan_preserves_dict_range_violations_within_reference_exception():
    # 31.58은 데이터사전 범위(8~30) 밖이지만 문헌 기반 예외(30~38.2) 안이라 원본 보존 + 플래그만 True
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 31.58, 25.0]})

    result = replace_outliers_with_nan(
        df,
        ranges={"Ni_함량_pct": (8, 30)},
        reference_exception_ranges={"Ni_함량_pct": ((30, 38.2),)},
    )

    assert result["Ni_함량_pct"].tolist() == [10.0, 31.58, 25.0]
    assert result["Ni_함량_pct_dict_range_flag"].tolist() == [False, True, False]


def test_replace_outliers_with_nan_does_not_flag_values_that_became_nan():
    # 999는 보존 기준 밖이라 NaN이 되고, 더 이상 "측정값"이 아니므로 플래그도 False여야 한다
    df = pd.DataFrame({"Ni_함량_pct": [999.0]})

    result = replace_outliers_with_nan(
        df,
        ranges={"Ni_함량_pct": (8, 30)},
        reference_exception_ranges={"Ni_함량_pct": ((30, 38.2),)},
    )

    assert np.isnan(result["Ni_함량_pct"].iloc[0])
    assert result["Ni_함량_pct_dict_range_flag"].iloc[0] == False


def test_outlier_history_distinguishes_missing_and_preserves_rejected_values():
    df = pd.DataFrame({"Ni_함량_pct": [np.nan, 999., np.inf, -np.inf, 31.58, 20.]})
    result = replace_outliers_with_nan(df)
    assert result["Ni_함량_pct_outlier_reason"].tolist() == [
        "original_missing", "outside_accepted_range", "non_finite",
        "non_finite", "not_replaced", "not_replaced",
    ]
    np.testing.assert_equal(
        result["Ni_함량_pct_outlier_original_value"].to_numpy(),
        [np.nan, 999., np.inf, -np.inf, np.nan, np.nan],
    )
    pd.testing.assert_frame_equal(replace_outliers_with_nan(result), result)
    assert df.loc[1, "Ni_함량_pct"] == 999.


def test_replace_outliers_with_nan_keeps_values_within_range_unchanged():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 30.0]})

    result = replace_outliers_with_nan(
        df, ranges={"Ni_함량_pct": (8, 30)}, reference_exception_ranges={"Ni_함량_pct": ((30, 38.2),)}
    )

    assert result["Ni_함량_pct"].tolist() == [10.0, 20.0, 30.0]
    assert result["Ni_함량_pct_dict_range_flag"].tolist() == [False, False, False]


def test_replace_outliers_with_nan_ignores_columns_not_in_dataframe():
    df = pd.DataFrame({"Ni_함량_pct": [10.0]})

    result = replace_outliers_with_nan(
        df,
        ranges={"존재하지않는컬럼": (0, 1)},
        reference_exception_ranges={},
    )

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


def test_imputation_rejects_acid_prediction_outside_range_with_batch_details():
    reference = pd.DataFrame([
        _full_row(20.0, 100.0, 600.0),
        _full_row(20.0, 150.0, 1400.0),
    ])
    target = pd.DataFrame([dict(_full_row(20.0, 200.0, None), 배치ID="B001")])
    with pytest.raises(ValueError, match="대치값") as error:
        impute_missing_from_reference(reference, target)
    assert "황산투입량_L" in str(error.value)
    assert "B001" in str(error.value)
    assert "2200" in str(error.value)
    assert target['황산투입량_L'].isna().all()


@pytest.mark.parametrize('column,value', [('Ni_함량_pct', 40.0), ('침출온도_C', 96.0), ('침출시간_hr', 9.0)])
def test_imputation_rejects_outside_range_medians(column, value):
    reference = pd.DataFrame([_full_row(20.0, 100.0, 700.0)])
    reference[column] = value
    target = pd.DataFrame([_full_row(20.0, 100.0, 700.0)], index=[7])
    target[column] = np.nan
    with pytest.raises(ValueError, match=column) as error:
        impute_missing_from_reference(reference, target)
    assert '행 인덱스' in str(error.value)
    assert '7' in str(error.value)


def test_imputation_accepts_reference_exception_and_ignores_existing_values():
    reference = pd.DataFrame([_full_row(38.2, 100.0, 700.0)])
    target = pd.DataFrame([
        _full_row(None, 100.0, 700.0),
        _full_row(999.0, 100.0, 700.0),
    ])
    result = impute_missing_from_reference(reference, target)
    assert result['Ni_함량_pct'].tolist() == [38.2, 999.0]


def test_prepare_split_applies_imputation_range_check():
    from module1_preprocessing.transformation import prepare_split
    reference = pd.DataFrame([
        _full_row(20.0, 100.0, 600.0),
        _full_row(20.0, 150.0, 1400.0),
    ])
    target = pd.DataFrame([dict(_full_row(20.0, 200.0, None), 배치ID="B001")])
    reference['분석일자'] = pd.Timestamp('2025-01-01')
    target['분석일자'] = pd.Timestamp('2025-01-02')
    with pytest.raises(ValueError, match='황산투입량_L'):
        prepare_split(reference, target)


@pytest.mark.parametrize('column,lower,upper', [
    ('Ni_함량_pct', 5.9, 38.2), ('Co_함량_pct', 2, 15),
    ('Li_함량_pct', 1.81, 6.76), ('Mn_함량_pct', 5, 16),
    ('Cu_불순물_pct', 0.05, 6.08), ('Fe_불순물_pct', 0.02, 6),
    ('Al_불순물_pct', 0.13, 7.46), ('원료투입량_kg', 80, 200),
    ('황산투입량_L', 400, 1500), ('침출온도_C', 40, 95),
    ('침출시간_hr', 1, 8),
])
def test_preservation_boundaries(column, lower, upper):
    df = pd.DataFrame({column: [lower, upper, lower - .001, upper + .001, np.inf, np.nan]})
    out = replace_outliers_with_nan(df)
    assert out[column].iloc[:2].tolist() == [lower, upper]
    assert out[column].iloc[2:].isna().all()
    assert check_ranges(df, reference_exception_ranges=REFERENCE_EXCEPTION_RANGES)[column] == [2, 3, 4]
    assert check_ranges(out, reference_exception_ranges=REFERENCE_EXCEPTION_RANGES) == {}
    lo, hi = NORMAL_RANGES[column]
    assert out[column + '_dict_range_flag'].tolist() == [lower < lo, upper > hi, False, False, False, False]


def test_mn_has_no_exception_and_targets_are_untouched():
    df = pd.DataFrame({'Mn_함량_pct': [16.02, 16.66, 16.92, 16.99],
                       'Ni_회수율_pct': [90, np.nan, 101, 92]})
    out = replace_outliers_with_nan(df)
    assert out['Mn_함량_pct'].isna().all()
    pd.testing.assert_series_equal(out['Ni_회수율_pct'], df['Ni_회수율_pct'])
    assert len(BASE_FEATURES) == len(set(BASE_FEATURES)) == 11
    assert not set(TARGET_COLUMNS) & set(BASE_FEATURES + DERIVED_FEATURES)
    assert DERIVED_FEATURES == ['acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'feed_impurity_g_per_l']


def test_disjoint_exception_does_not_admit_gap():
    df = pd.DataFrame({'x': [1, 3, 5]})
    out = replace_outliers_with_nan(df, ranges={'x': (0, 2)}, reference_exception_ranges={'x': ((4, 6),)})
    assert out['x'].iloc[0] == 1
    assert pd.isna(out['x'].iloc[1])
    assert out['x'].iloc[2] == 5
