import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """배치ID 순번(분석일자 순서의 근사)으로 시간순 정렬한 뒤 Train/Validation/Test
    구간으로 고정 분할한다 (무작위 섞기 없음 — 미래 배치로 과거를 평가하지 않도록).
    """
    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    ordered_index = batch_seq.sort_values().index

    n = len(ordered_index)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return ordered_index[:train_end], ordered_index[train_end:val_end], ordered_index[val_end:]
