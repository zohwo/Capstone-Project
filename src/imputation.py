import numpy as np
import pandas as pd

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


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """조성 변수는 원료등급 그룹별 중앙값, 공정조건은 전체 중앙값으로 대치한다."""
    df = df.copy()

    for col in COMPOSITION_COLUMNS:
        df[col] = df.groupby("원료등급")[col].transform(lambda s: s.fillna(s.median()))

    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    return df
