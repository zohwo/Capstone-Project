import numpy as np
import pandas as pd
import pytest

from module2_prediction.artifacts import load_bundle, predict_bundle
from module2_prediction.evaluation import evaluate_final_model
from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS
from module2_prediction.tests.test_tuning import _make_fixture


@pytest.mark.parametrize("column,value", [
    ("Ni_함량_pct", -10), ("Co_함량_pct", 101),
    ("Li_함량_pct", np.inf), ("Mn_함량_pct", -np.inf),
    ("Ni_함량_pct", "invalid"),
    ("원료투입량_kg", 0), ("원료투입량_kg", 201), ("원료투입량_kg", np.nan),
    ("황산투입량_L", 399), ("황산투입량_L", 1501),
    ("침출온도_C", 39), ("침출온도_C", 96),
    ("침출시간_hr", 0), ("침출시간_hr", 9),
])
def test_predict_bundle_rejects_invalid_input_before_prediction(column, value):
    frame = _make_fixture().iloc[[0]].copy()
    frame[column] = value
    with pytest.raises(ValueError, match=column):
        predict_bundle({}, frame)


def test_predict_bundle_accepts_boundaries_and_preserves_missing_imputation():
    from module1_preprocessing.config import COMPOSITION_COLUMNS, NORMAL_RANGES

    class Recorder:
        def predict(self, values):
            assert np.isfinite(values.to_numpy(dtype=float)).all()
            return values['Ni_함량_pct'].to_numpy()

    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS, model=Recorder())
    frame = reference.iloc[:2].copy()
    # 조성 7종 합계는 100%를 넘을 수 없음 — Ni_함량_pct만 경계값(0/100)으로 두고 나머지는 0
    for col in COMPOSITION_COLUMNS:
        frame[col] = 0.0
    frame['Ni_함량_pct'] = [0., 100.]
    for col in ['원료투입량_kg', '황산투입량_L', '침출온도_C', '침출시간_hr']:
        frame[col] = list(NORMAL_RANGES[col])
    np.testing.assert_equal(predict_bundle(bundle, frame), [0., 100.])
    missing = reference.iloc[[0]].copy()
    missing[['Ni_함량_pct', '황산투입량_L', '침출온도_C', '침출시간_hr']] = np.nan
    expected = reference.groupby('원료등급')['Ni_함량_pct'].median()['A']
    np.testing.assert_allclose(predict_bundle(bundle, missing), [expected])


def test_predict_bundle_rejects_composition_sum_over_100_percent():
    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS, model=None)
    frame = reference.iloc[[0]].copy()
    frame['Ni_함량_pct'] = 80.0
    frame['Co_함량_pct'] = 60.0  # 두 컬럼 다 개별로는 0~100% 안이지만 합계 140% > 100%

    with pytest.raises(ValueError, match='100%'):
        predict_bundle(bundle, frame)


def test_predict_bundle_rejects_missing_column():
    frame = _make_fixture().iloc[[0]].drop(columns='황산투입량_L')
    with pytest.raises(ValueError, match='황산투입량_L'):
        predict_bundle({}, frame)


def test_predict_bundle_rejects_grade_missing_only_on_rows_that_need_it():
    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS, model=None)
    frame = reference.iloc[:2].copy()
    frame.loc[frame.index[0], 'Ni_함량_pct'] = np.nan
    frame['원료등급'] = pd.NA

    with pytest.raises(ValueError, match=str(frame.index[0])) as exc:
        predict_bundle(bundle, frame)
    assert str(frame.index[1]) not in str(exc.value)


def test_predict_bundle_rejects_blank_string_grade():
    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS, model=None)
    frame = reference.iloc[[0]].copy()
    frame['Ni_함량_pct'] = np.nan
    frame['원료등급'] = '   '

    with pytest.raises(ValueError, match='원료등급'):
        predict_bundle(bundle, frame)


def test_predict_bundle_requires_grade_for_all_rows_when_grade_is_model_feature():
    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS + ['원료등급'], model=None)

    frame = reference.iloc[[0]].copy()
    frame['원료등급'] = pd.NA

    with pytest.raises(ValueError, match='원료등급'):
        predict_bundle(bundle, frame)


def test_predict_bundle_allows_missing_grade_column_when_not_required():
    class Recorder:
        def predict(self, values):
            return values['Ni_함량_pct'].to_numpy()

    reference = _make_fixture()
    bundle = dict(reference=reference, features=FEATURE_COLUMNS, model=Recorder())
    frame = reference.iloc[[0]].drop(columns='원료등급')

    result = predict_bundle(bundle, frame)
    assert np.isfinite(result).all()


def test_saved_model_reuses_development_statistics_without_targets(tmp_path):
    data = _make_fixture()
    from module1_preprocessing.config import NORMAL_RANGES
    
    for col in ['침출온도_C', '침출시간_hr']:
        data[col] = data[col].clip(*NORMAL_RANGES[col])
    train, val, test = data.iloc[:30].copy(), data.iloc[30:45].copy(), data.iloc[45:].copy()
    train.loc[0, 'Ni_함량_pct'] = np.nan
    test.iloc[0, test.columns.get_loc('Ni_함량_pct')] = np.nan
    target = TARGET_COLUMNS[0]
    evaluate_final_model(train, val, test, 'RandomForest', {'n_estimators': 5},
                         target_columns=[target], output_dir=tmp_path)
    bundle = load_bundle(tmp_path / f'{target}.joblib')
    assert not set(TARGET_COLUMNS).intersection(bundle['reference'].columns)
    assert len(bundle['reference']) == 45
    assert bundle['features'] == FEATURE_COLUMNS
    actual = predict_bundle(bundle, test.drop(columns=TARGET_COLUMNS))
    expected = pd.read_csv(tmp_path / f'{target}_predictions.csv')['prediction']
    np.testing.assert_allclose(actual, expected)
    changed = test.copy()
    # 원래 평균(~20)과 다르지만, 조성 합계 100% 제약을 지키는 값으로 변경
    changed.iloc[1:, changed.columns.get_loc('Ni_함량_pct')] = 10
    assert predict_bundle(bundle, changed)[0] == actual[0]
    complete = test.iloc[1:].drop(columns=TARGET_COLUMNS + ['원료등급', '분석일자'])
    np.testing.assert_allclose(predict_bundle(bundle, complete), actual[1:])
