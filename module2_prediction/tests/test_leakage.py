"""모듈 간 연결과 CV의 정보 경계에 대한 회귀 테스트."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from module1_preprocessing.config import COMPOSITION_COLUMNS
from module1_preprocessing.pipeline import run as preprocess
from module2_prediction import evaluation, pipeline, tuning
from module2_prediction.config import TARGET_COLUMNS
from module2_prediction.data_loading import load_modeling_inputs


def raw_frame():
    n = 30
    bounds = [(10, 25), (3, 10), (2, 5), (5, 12), (0.5, 3), (0.5, 3), (0.5, 3)]
    df = pd.DataFrame({c: np.linspace(lo, hi, n) for c, (lo, hi) in zip(COMPOSITION_COLUMNS, bounds)})
    df['원료등급'] = ['A', 'B', 'C'] * 10
    df['배치ID'] = [f'BP-2025-{i:04d}' for i in range(n)]
    df['분석일자'] = pd.Timestamp('2025-01-01')
    df['침출온도_C'] = 60.0
    df['침출시간_hr'] = 3.0
    df['원료투입량_kg'] = 100.0
    df['황산투입량_L'] = 600.0
    for target in TARGET_COLUMNS:
        df[target] = np.arange(n) + 60.0
    return df


def test_cv_heldout_changes_cannot_change_training_imputation(monkeypatch):
    df = raw_frame()
    tr, va = next(StratifiedKFold(3, shuffle=True, random_state=42).split(df, df['원료등급']))
    df.loc[tr[0], 'Ni_함량_pct'] = np.nan
    fits = []

    class Recorder:
        def fit(self, X, y):
            fits.append(X.copy())
        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(tuning, 'MODEL_FACTORIES', {'Recorder': lambda **kw: Recorder()})
    monkeypatch.setattr(tuning, 'MODEL_HYPERPARAMETER_GRIDS', {'Recorder': [{}]})
    features = {TARGET_COLUMNS[0]: ['Ni_함량_pct']}
    tuning.tune_models_cv(df, feature_columns=features, target_columns=[TARGET_COLUMNS[0]], n_splits=3)
    before = fits[0]
    changed = df.copy()
    changed.loc[va, 'Ni_함량_pct'] = 30.0
    fits.clear()
    tuning.tune_models_cv(changed, feature_columns=features, target_columns=[TARGET_COLUMNS[0]], n_splits=3)
    pd.testing.assert_frame_equal(before, fits[0])
    assert before.iloc[0, 0] == df.iloc[tr].groupby('원료등급')['Ni_함량_pct'].median()[df.loc[tr[0], '원료등급']]


def test_tuning_uses_rmse_even_when_mae_and_r2_prefer_other_candidate(monkeypatch):
    monkeypatch.setattr(tuning, 'MODEL_FACTORIES', {'Fake': lambda **params: params})
    monkeypatch.setattr(tuning, 'MODEL_HYPERPARAMETER_GRIDS', {'Fake': [{'candidate': 0}, {'candidate': 1}]})
    def score(model, *args):
        return {'train_rmse': 0.1, 'val_r2': [0.99, 0.5][model['candidate']], 'val_mse': 4.0,
                'val_rmse': [2.0, 1.0][model['candidate']], 'val_mae': [0.5, 0.9][model['candidate']]}
    monkeypatch.setattr(tuning, '_evaluate_prepared', score)
    df = raw_frame()
    result = tuning.tune_models_cv(df, n_splits=3)
    assert all(v['params']['candidate'] == 1 for v in result.values())


def test_tuning_rejects_test_rows():
    df = raw_frame()
    df['split'] = 'train'
    df.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError, match='Test'):
        tuning.tune_models_cv(df)


def test_final_imputation_uses_development_not_test(monkeypatch):
    df = raw_frame()
    df['원료등급'] = 'A'
    train, val, test = df.iloc[:10].copy(), df.iloc[10:20].copy(), df.iloc[20:].copy()
    train['Ni_함량_pct'] = 10.0
    val['Ni_함량_pct'] = 30.0
    train.loc[0, 'Ni_함량_pct'] = np.nan
    test['Ni_함량_pct'] = 99.0
    test.iloc[0, test.columns.get_loc('Ni_함량_pct')] = np.nan
    captured = {}
    class Recorder:
        def fit(self, X, y):
            captured['train'] = X.copy()
        def predict(self, X):
            captured['test'] = X.copy()
            return np.arange(len(X)) + 70.0
    monkeypatch.setitem(evaluation.MODEL_FACTORIES, 'Recorder', lambda **kw: Recorder())
    evaluation.evaluate_final_model(train, val, test, 'Recorder', {},
                                   feature_columns=['Ni_함량_pct'], target_columns=[TARGET_COLUMNS[0]])
    assert captured['train'].iloc[0, 0] == 30.0
    assert captured['test'].iloc[0, 0] == 30.0


def test_module1_artifacts_preserve_missing_and_bind_selection(tmp_path, capsys):
    from pathlib import Path
    preprocess(Path('data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx'), tmp_path / 'cleaned_features.csv')
    output = capsys.readouterr().out
    assert '최종 Test 성능' not in output
    df, selection = load_modeling_inputs(tmp_path / 'pre_impute_data.csv')
    assert df[COMPOSITION_COLUMNS].isna().any().any()
    assert not pd.read_csv(tmp_path / 'cleaned_features.csv')[COMPOSITION_COLUMNS].isna().any().any()
    assert selection['primary_metric'] == 'mae'
    assert set(selection['targets']) == set(TARGET_COLUMNS)
    for target, entry in selection['targets'].items():
        assert target not in entry['features']
    with pytest.raises(ValueError, match='일치'):
        load_modeling_inputs(tmp_path / 'cleaned_features.csv')
    df.loc[0, 'Ni_함량_pct'] = 99.0
    df.to_csv(tmp_path / 'pre_impute_data.csv', index=False)
    with pytest.raises(ValueError, match='일치'):
        load_modeling_inputs(tmp_path / 'pre_impute_data.csv')


def test_pipeline_forwards_target_features_and_selects_rmse(monkeypatch):
    df = raw_frame()
    df['split'] = ['train'] * 15 + ['validation'] * 6 + ['test'] * 9
    features = {TARGET_COLUMNS[0]: ['Ni_함량_pct'], TARGET_COLUMNS[1]: ['Co_함량_pct', 'leach_intensity']}
    selection = {'targets': {t: {'features': f} for t, f in features.items()},
                 'cv': {'n_splits': 3, 'random_state': 42, 'stratify_column': '원료등급'}}
    monkeypatch.setattr(pipeline, 'load_modeling_inputs', lambda *args: (df, selection))
    def tune(pool, feature_columns, **kw):
        assert set(pool['split']) == {'train', 'validation'}
        assert feature_columns == features
        return {(name, target): {'params': {}, 'metrics': {'val_mae': 3-mae, 'val_rmse': mae, 'val_r2': r2}}
                for target in TARGET_COLUMNS for name, mae, r2 in [('RMSE', 1., .5), ('R2', 2., .9)]}
    def evaluate(train, val, test, model_name, model_params, target_columns, feature_columns, output_dir=None):
        assert model_name == 'RMSE'
        assert feature_columns == features[target_columns[0]]
        return {target_columns[0]: {'test_mae': 1.0}}
    monkeypatch.setattr(pipeline, 'tune_models_cv', tune)
    monkeypatch.setattr(pipeline, 'evaluate_final_model', evaluate)
    result = pipeline.run('unused')
    assert all(v['model'] == 'RMSE' for v in result.values())


@pytest.mark.parametrize("value", [-1., 999., np.nan, np.inf])
def test_target_validation_rejects_invalid_truth(value):
    from module1_preprocessing.transformation import validate_targets
    df = raw_frame()
    df.loc[0, TARGET_COLUMNS[0]] = value
    with pytest.raises(ValueError, match="BP-2025-0000"):
        validate_targets(df, TARGET_COLUMNS)
    with pytest.raises(ValueError, match="목표변수"):
        tuning.tune_models_cv(df, n_splits=3)


def test_base_contains_all_raw_information():
    from module1_preprocessing.config import BASE_FEATURES, DERIVED_FEATURES
    from module2_prediction.config import FEATURE_COLUMNS
    assert len(BASE_FEATURES) == 11
    assert {"원료투입량_kg", "황산투입량_L"}.issubset(BASE_FEATURES)
    assert FEATURE_COLUMNS == BASE_FEATURES + DERIVED_FEATURES


def test_synthetic_pipeline_round_trip_and_test_isolation(tmp_path, monkeypatch):
    from module1_preprocessing.pipeline import select_and_save_features
    from module1_preprocessing.config import BASE_FEATURES
    source = raw_frame()
    source.loc[0, 'Ni_함량_pct'] = np.nan
    source['분석일자'] = '2025-01-01'
    source_path = tmp_path / 'source.csv'
    source.to_csv(source_path, index=False)
    preprocess(source_path, tmp_path / 'cleaned_features.csv')
    original, selected = load_modeling_inputs(tmp_path / 'pre_impute_data.csv')
    assert selected['schema_version'] == 2
    assert original['Ni_함량_pct'].isna().sum() == 1
    assert all(set(BASE_FEATURES).issubset(v['features']) for v in selected['targets'].values())
    assert not pd.read_csv(tmp_path / 'cleaned_features.csv')['Ni_함량_pct'].isna().any()
    changed = original.copy()
    changed.loc[changed['split'] == 'test', TARGET_COLUMNS] = 99.
    changed.loc[changed['split'] == 'test', 'Ni_함량_pct'] = 50.
    updated = select_and_save_features(changed, tmp_path / 'other' / 'cleaned_features.csv')
    assert selected['targets'] == updated['targets']
    assert selected['data_sha256'] != updated['data_sha256']
    monkeypatch.setattr(tuning, 'MODEL_HYPERPARAMETER_GRIDS', {'RandomForest': [{'n_estimators': 5}]})
    result = pipeline.run(tmp_path / 'pre_impute_data.csv')
    assert set(result) == set(TARGET_COLUMNS)
    assert all(np.isfinite(v['test_mae']) for v in result.values())
