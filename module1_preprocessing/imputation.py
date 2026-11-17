import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from module1_preprocessing.config import (
    NORMAL_RANGES,
    REFERENCE_EXCEPTION_RANGES,
    COMPOSITION_COLUMNS,
    PROCESS_COLUMNS,
)


def accepted_range_mask(values: pd.Series, normal_range: tuple, exceptions: tuple = ()) -> pd.Series:
    """정상범위 또는 명시한 예외 구간에 속하는 유한한 값인지 판정"""
    accepted = values.between(*normal_range)
    for low, high in exceptions:
        accepted |= values.between(low, high)
    return (accepted & np.isfinite(values)).fillna(False)


def replace_outliers_with_nan(
    df: pd.DataFrame,
    ranges: dict = NORMAL_RANGES,
    reference_exception_ranges: dict = REFERENCE_EXCEPTION_RANGES,
) -> pd.DataFrame:
    """정상범위와 문헌 기반 예외 밖의 값을 NaN으로 변환

    정상범위 밖에서 예외로 보존된 값만 '{컬럼}_dict_range_flag'로 표시
    '{컬럼}_outlier_reason'에 original_missing / not_replaced /
    outside_accepted_range / non_finite를 기록하고, 결측으로 바꾼 값은
    '{컬럼}_outlier_original_value'에 보존한다. 이력은 모델 피처가 아니다.
    """
    df = df.copy()
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue
        accepted = accepted_range_mask(df[col], bounds, reference_exception_ranges.get(col, ()))
        rejected = df[col].notna() & ~accepted
        reason_col = f"{col}_outlier_reason"
        original_col = f"{col}_outlier_original_value"
        # 재호출 시 이미 기록된 이력을 원래 결측으로 덮어쓰지 않는다.
        if reason_col not in df:
            df[reason_col] = pd.Series("not_replaced", index=df.index, dtype="string")
            df.loc[df[col].isna(), reason_col] = "original_missing"
        if original_col not in df:
            df[original_col] = np.nan
        df.loc[rejected, original_col] = df.loc[rejected, col]
        df.loc[rejected, reason_col] = "outside_accepted_range"
        df.loc[rejected & ~np.isfinite(df[col]), reason_col] = "non_finite"
        df.loc[~accepted, col] = np.nan
        df[f"{col}_dict_range_flag"] = (df[col].notna() & ~df[col].between(*bounds)).fillna(False)
    return df


def interpolate_analysis_date(df: pd.DataFrame) -> pd.DataFrame:
    """배치ID의 숫자 순번을 시간 순서로 가정해 분석일자 결측값을 선형 보간"""

    df = df.copy()

    if "배치ID" not in df.columns or "분석일자" not in df.columns:
        return df

    if df["분석일자"].notna().sum() == 0:
        return df

    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    ordered_index = batch_seq.sort_values().index

    dates = df.loc[ordered_index, "분석일자"].astype("datetime64[ns]")

    numeric = dates.to_numpy().astype("int64").astype("float64")
    numeric[dates.isna().to_numpy()] = np.nan

    interpolated = pd.Series(
        numeric,
        index=ordered_index
    ).interpolate(limit_direction="both")

    df.loc[ordered_index, "분석일자"] = pd.to_datetime(
        interpolated.round().astype("int64"),
        unit="ns"
    )

    return df


def impute_missing_from_reference(
    reference_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """reference_df에서 계산한 통계와 회귀관계로 target_df의 결측치를 대치

    - 조성 변수: 원료등급별 중앙값으로 대치하고, 해당 등급 통계가 없으면 전체 중앙값을 사용
    - 공정 변수: reference_df의 전체 중앙값으로 대치
    - 황산투입량: 원료투입량과의 선형회귀 관계를 이용해 대치
    - Validation/Test 대치 시 Train 데이터만 reference_df로 사용해 데이터 누수를 방지
    """

    out = target_df.copy()

    for col in COMPOSITION_COLUMNS:
        group_medians = reference_df.groupby("원료등급")[col].median()
        overall_median = reference_df[col].median()

        missing = out[col].isna()

        if missing.any():
            out.loc[missing, col] = (
                out.loc[missing, "원료등급"].map(group_medians).fillna(overall_median)
            )

    for col in PROCESS_COLUMNS:
        out[col] = out[col].fillna(reference_df[col].median())

    ref_acid = reference_df[["원료투입량_kg", "황산투입량_L"]].dropna()

    if not ref_acid.empty:
        model = LinearRegression().fit(ref_acid[["원료투입량_kg"]], ref_acid["황산투입량_L"])

        acid_missing = out["황산투입량_L"].isna() & out["원료투입량_kg"].notna()

        if acid_missing.any():
            out.loc[acid_missing, "황산투입량_L"] = model.predict(
                out.loc[acid_missing, ["원료투입량_kg"]]
            )

    violations = {}
    for col in COMPOSITION_COLUMNS + PROCESS_COLUMNS + ["황산투입량_L"]:
        filled = target_df[col].isna() & out[col].notna()
        if not filled.any():
            continue
        filled_rows = out.loc[filled]
        invalid = ~accepted_range_mask(
            filled_rows[col].astype(float), NORMAL_RANGES[col], REFERENCE_EXCEPTION_RANGES.get(col, ())
        )
        if invalid.any():
            rows = filled_rows.loc[invalid, ["배치ID", col]] if "배치ID" in out else (
                filled_rows.loc[invalid, [col]].rename_axis("행 인덱스").reset_index()
            )
            violations[col] = rows.to_dict(orient="records")
    if violations:
        raise ValueError(f"대치값이 정상범위 또는 문헌 기반 보존 예외를 벗어남: {violations}")

    return out
