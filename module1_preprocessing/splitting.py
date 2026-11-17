import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    stratify_column: str = "원료등급",
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    stratify_column의 클래스 비율을 유지하며 무작위로 Train/Validation/Test로 분할
    """

    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac과 val_frac은 양수이며 합이 1보다 작아야 합니다.")

    test_frac = 1 - train_frac - val_frac
    test_count = round(len(df) * test_frac)
    val_count = round(len(df) * val_frac)
    if min(test_count, val_count, len(df) - test_count - val_count) < 1:
        raise ValueError("분할 비율과 표본 수로 각 세트에 최소 1개 행을 배정할 수 없습니다.")

    development_i, test_i = train_test_split(
        df.index,
        test_size=test_count,
        stratify=df[stratify_column],
        random_state=random_state,
    )
    train_i, val_i = train_test_split(
        development_i,
        test_size=val_count,
        stratify=df.loc[development_i, stratify_column],
        random_state=random_state,
    )

    return pd.Index(train_i), pd.Index(val_i), pd.Index(test_i)
