import pandas as pd

from src.splitting import temporal_train_val_test_split


def _make_batch_df(n=40):
    return pd.DataFrame({"배치ID": [f"BP-2025-{i:04d}" for i in range(n)]})


def test_temporal_split_respects_batch_order_without_shuffling():
    df = _make_batch_df(n=40)

    train_i, val_i, test_i = temporal_train_val_test_split(df, train_frac=0.7, val_frac=0.15)

    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    assert batch_seq.loc[train_i].max() < batch_seq.loc[val_i].min()
    assert batch_seq.loc[val_i].max() < batch_seq.loc[test_i].min()
    assert len(train_i) + len(val_i) + len(test_i) == len(df)


def test_temporal_split_sizes_match_requested_fractions():
    df = _make_batch_df(n=40)

    train_i, val_i, test_i = temporal_train_val_test_split(df, train_frac=0.7, val_frac=0.15)

    assert len(train_i) == 28
    assert len(val_i) == 6
    assert len(test_i) == 6


def test_temporal_split_works_on_unsorted_input():
    df = _make_batch_df(n=10)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    train_i, val_i, test_i = temporal_train_val_test_split(shuffled, train_frac=0.7, val_frac=0.2)

    batch_seq = shuffled["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    assert batch_seq.loc[train_i].max() < batch_seq.loc[val_i].min()
    assert batch_seq.loc[val_i].max() < batch_seq.loc[test_i].min()
