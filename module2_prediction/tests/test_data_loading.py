import pandas as pd
import pytest

from module2_prediction.data_loading import load_cleaned_features, split_by_column


def test_load_cleaned_features_reads_csv(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"a": [1, 2], "split": ["train", "test"]}).to_csv(csv_path, index=False)

    df = load_cleaned_features(csv_path)

    assert df.shape == (2, 2)


def test_split_by_column_separates_train_validation_test():
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 6],
        "split": ["train", "train", "validation", "validation", "test", "test"],
    })

    train_df, val_df, test_df = split_by_column(df)

    assert train_df["value"].tolist() == [1, 2]
    assert val_df["value"].tolist() == [3, 4]
    assert test_df["value"].tolist() == [5, 6]


def test_split_by_column_raises_without_split_column():
    df = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(ValueError):
        split_by_column(df)
