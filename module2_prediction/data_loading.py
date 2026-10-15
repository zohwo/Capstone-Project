import pandas as pd

from module2_prediction.config import CLEANED_DATA_PATH


def load_cleaned_features(path=CLEANED_DATA_PATH) -> pd.DataFrame:
    """module1_preprocessing의 산출물(cleaned_features.csv)을 읽는다."""
    return pd.read_csv(path)


def split_by_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """df의 `split` 컬럼(train/validation/test) 기준으로 세 DataFrame으로 나눈다."""
    if "split" not in df.columns:
        raise ValueError("`split` 컬럼이 없습니다 — module1_preprocessing의 산출물이 맞는지 확인하세요.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df
