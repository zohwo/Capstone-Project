import pandas as pd
import pytest

from module2_prediction.data_loading import (
    load_batch_data,
    split_by_column,
)


def test_load_batch_data_reads_csv(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({
        "a": [1, 2], "배치ID": ["BP-0000", "BP-0001"], "split": ["train", "test"],
    }).to_csv(csv_path, index=False)

    df = load_batch_data(csv_path)

    assert df.shape == (2, 3)


def test_load_batch_data_raises_without_batch_id_column(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"a": [1, 2], "split": ["train", "test"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="배치ID"):
        load_batch_data(csv_path)


def test_load_batch_data_strips_whitespace_from_batch_id(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({
        "a": [1, 2], "배치ID": [" BP-0000 ", "BP-0001"], "split": ["train", "test"],
    }).to_csv(csv_path, index=False)

    df = load_batch_data(csv_path)

    assert df["배치ID"].tolist() == ["BP-0000", "BP-0001"]


def test_load_batch_data_raises_on_empty_batch_id(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({
        "a": [1, 2], "배치ID": ["BP-0000", "  "], "split": ["train", "test"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="빈 문자열"):
        load_batch_data(csv_path)


def test_load_batch_data_raises_on_duplicate_batch_id_across_splits(tmp_path):
    """같은 배치가 Train과 Test에 동시에 들어가면 새 배치에 대한 예측 성능이
    과대추정될 수 있다 — 1행=1배치 가정이므로 전역 유일성만 확인하면 된다."""
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({
        "a": [1, 2, 3], "배치ID": ["BP-0142", "BP-0001", "BP-0142"],
        "split": ["train", "validation", "test"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="BP-0142"):
        load_batch_data(csv_path)


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


def test_split_by_column_accepts_non_temporal_label_order():
    """module1이 stratified(원료등급 비율 유지) 분할로 전환하면서 split 라벨이
    더 이상 train→validation→test 블록 순서로 오지 않는다 — tabular 모델은 행을
    독립적으로 다루므로 이건 정상이고, split_by_column이 막으면 안 된다."""
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 6],
        "배치ID": [f"BP-{i:04d}" for i in range(6)],
        "split": ["test", "test", "train", "train", "validation", "validation"],
    })

    train_df, val_df, test_df = split_by_column(df)

    assert train_df["value"].tolist() == [3, 4]
    assert val_df["value"].tolist() == [5, 6]
    assert test_df["value"].tolist() == [1, 2]


def test_load_batch_data_preserves_numeric_ids(tmp_path):
    path = tmp_path / 'batches.csv'
    path.write_text('배치ID,value\n001,1\n01,2\n1,3\n')
    df = load_batch_data(path)
    assert df['배치ID'].tolist() == ['001', '01', '1']
    assert isinstance(df['배치ID'].dtype, pd.StringDtype)


@pytest.mark.parametrize('location', ['root', 'targets', 'entry', 'cv'])
@pytest.mark.parametrize('invalid', [None, [], 'invalid', 1])
def test_modeling_loader_rejects_non_object_settings(tmp_path, location, invalid):
    import hashlib
    import json
    from module2_prediction.config import TARGET_COLUMNS
    from module2_prediction.data_loading import load_modeling_inputs
    from module2_prediction.tests.test_leakage import raw_frame

    frame = raw_frame()  # 합성 입력이며 원본 파일에 접근하지 않는다.
    frame['split'] = ['train'] * 18 + ['validation'] * 6 + ['test'] * 6
    data_path = tmp_path / 'pre_impute_data.csv'
    frame.to_csv(data_path, index=False)
    settings = dict(schema_version=2, preprocessing_stage='before_imputation', primary_metric='mae',
                    data_file=data_path.name, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    targets={t: {'features': ['Ni_함량_pct']} for t in TARGET_COLUMNS},
                    cv=dict(n_splits=3, shuffle=True, random_state=42, stratify_column='원료등급'))
    expected = {'root': '최상위', 'targets': 'targets', 'entry': TARGET_COLUMNS[0], 'cv': 'cv'}[location]
    if location == 'root':
        settings = invalid
    elif location == 'entry':
        settings['targets'][TARGET_COLUMNS[0]] = invalid
    else:
        settings[location] = invalid
    (tmp_path / 'feature_selection.json').write_text(json.dumps(settings), encoding='utf-8')
    with pytest.raises(ValueError, match=expected):
        load_modeling_inputs(data_path)
