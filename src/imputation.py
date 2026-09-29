import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import NORMAL_RANGES, COMPOSITION_COLUMNS, PROCESS_COLUMNS


def flag_outliers(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> pd.DataFrame:
    """데이터사전 정상 범위를 벗어난 값을 NaN으로 치환한다 (이상치→결측 변환)."""
    df = df.copy()

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < low) | (df[col] > high)
        df.loc[out_of_range, col] = np.nan

    return df


def impute_acid_volume(df: pd.DataFrame) -> pd.DataFrame:
    """황산투입량_L 결측을 원료투입량_kg 기반 선형회귀(산-광석비)로 대치한다."""
    df = df.copy()

    if "황산투입량_L" not in df.columns or "원료투입량_kg" not in df.columns:
        return df

    missing_mask = df["황산투입량_L"].isna()
    if not missing_mask.any():
        return df

    train = df.loc[~missing_mask, ["원료투입량_kg", "황산투입량_L"]].dropna()
    model = LinearRegression().fit(train[["원료투입량_kg"]], train["황산투입량_L"])

    df.loc[missing_mask, "황산투입량_L"] = model.predict(
        df.loc[missing_mask, ["원료투입량_kg"]]
    )

    return df


def interpolate_analysis_date(df: pd.DataFrame) -> pd.DataFrame:
    """배치ID의 순번을 시간 순서 근사값으로 사용해 분석일자 결측을 보간한다."""
    df = df.copy()

    if "배치ID" not in df.columns or "분석일자" not in df.columns:
        return df

    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    ordered_index = batch_seq.sort_values().index

    # Force nanosecond resolution explicitly: pandas' default datetime64
    # resolution has changed across versions (e.g. pandas 3.x may store
    # parsed dates in a coarser unit than pandas 2.x), and to_numpy().astype
    # ("int64") reads whatever unit the Series actually has. Without pinning
    # both the read and the write to "ns", a resolution mismatch silently
    # reinterprets the integer timestamp in the wrong unit (e.g. 2025 dates
    # come back as 1970).
    dates = df.loc[ordered_index, "분석일자"].astype("datetime64[ns]")
    numeric = dates.to_numpy().astype("int64").astype("float64")
    numeric[dates.isna().to_numpy()] = float("nan")

    interpolated = pd.Series(numeric, index=ordered_index).interpolate(
        limit_direction="both"
    )
    df.loc[ordered_index, "분석일자"] = pd.to_datetime(
        interpolated.round().astype("int64"), unit="ns"
    )

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """조성 변수는 원료등급 그룹별 중앙값, 공정조건은 전체 중앙값으로 대치한다."""
    df = df.copy()

    for col in COMPOSITION_COLUMNS:
        df[col] = df.groupby("원료등급")[col].transform(lambda s: s.fillna(s.median()))

    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    df = impute_acid_volume(df)
    df = interpolate_analysis_date(df)

    return df
