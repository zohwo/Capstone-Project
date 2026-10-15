import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """배치ID의 숫자 순번을 시간 순서의 근사값으로 사용해 Train/Validation/Test로 분할한다.
    무작위로 섞지 않고 순서대로 고정 분할해 미래 배치 정보가 과거 배치 학습에 포함되지 않도록 한다.
    """
    
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac과 val_frac은 양수이며 합이 1보다 작아야 합니다.")

    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    ordered_index = batch_seq.sort_values().index

    n = len(ordered_index)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return ordered_index[:train_end], ordered_index[train_end:val_end], ordered_index[val_end:]
