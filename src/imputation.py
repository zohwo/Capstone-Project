import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import NORMAL_RANGES, COMPOSITION_COLUMNS, PROCESS_COLUMNS


def replace_outliers_with_nan(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> pd.DataFrame:
    """데이터사전 정상 범위를 벗어난 값을 NaN으로 치환한다 (이상치→결측 변환)."""

    df = df.copy()

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue

        out_of_range = (df[col] < low) | (df[col] > high)
        df.loc[out_of_range, col] = np.nan

    return df

def interpolate_analysis_date(df: pd.DataFrame) -> pd.DataFrame:
    """배치ID의 숫자 순번을 시간 순서로 가정해 분석일자 결측값을 선형 보간한다."""

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
    """reference_df에서 계산한 통계와 회귀관계로 target_df의 결측치를 대치한다.

    - 조성 변수: 원료등급별 중앙값으로 대치하고, 해당 등급 통계가 없으면 전체 중앙값을 사용한다.
    - 공정 변수: reference_df의 전체 중앙값으로 대치한다.
    - 황산투입량: 원료투입량과의 선형회귀 관계를 이용해 대치한다.
    - Validation/Test 대치 시 Train 데이터만 reference_df로 사용해 데이터 누수를 방지한다.
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

    return out
