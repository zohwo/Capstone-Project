import numpy as np
import pandas as pd

from src.config import NORMAL_RANGES


def flag_outliers(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> pd.DataFrame:
    """데이터사전 정상 범위를 벗어난 값을 NaN으로 치환한다 (이상치→결측 변환)."""
    df = df.copy()

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < low) | (df[col] > high)
        df.loc[out_of_range, col] = np.nan

    return df
