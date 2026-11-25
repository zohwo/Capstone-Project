import pandas as pd
import pytest

from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS
from module2_prediction.scripts import oof_errors as experiment
from module2_prediction.scripts.oof_errors import generate_oof, validate_oof


def synthetic_cv_pool():
    rows = []
    for grade in ['A', 'B']:
        for i in range(3):
            rows.append(dict(
                배치ID=f'{grade}{i}', 원료등급=grade, 원료투입량_kg=140.0, 황산투입량_L=900.0,
                분석일자=pd.NaT, Ni_회수율_pct=80.0 + i, Co_회수율_pct=75.0 + i,
                **{c: 10.0 + i for c in COMPOSITION_COLUMNS},
                **{c: 70.0 + i for c in PROCESS_COLUMNS},
            ))
    return pd.DataFrame(rows)


def results_config():
    params = dict(alpha=0.1, l1_ratio=0.5)
    return dict(Ni_회수율_pct=dict(model='ElasticNet', params=params, features=['Ni_함량_pct']),
                Co_회수율_pct=dict(model='ElasticNet', params=params, features=['Co_함량_pct']))


def test_generate_oof_covers_input_pool_once_per_target_with_shared_folds():
    """generate_oof()가 입력 풀의 배치만, 타깃당 정확히 한 번, Ni·Co 공유 fold로 반환하는지 확인

    main()이 Train·Validation만 이 함수에 넘기는지는 여기서 확인하지 않음 — 그건
    module2_prediction.data_loading.split_by_column의 책임이고, 그쪽 테스트에서 다룸
    """
    cv_pool = synthetic_cv_pool()
    cv_config = dict(n_splits=2, random_state=0, stratify_column='원료등급')
    oof = generate_oof(cv_pool, cv_config, results_config())
    assert len(oof) == len(cv_pool) * 2
    assert oof.groupby('target')['배치ID'].apply(lambda s: s.is_unique).all()
    assert set(oof['배치ID']) == set(cv_pool['배치ID'])
    fold_by_batch = oof.pivot(index='배치ID', columns='target', values='fold')
    assert (fold_by_batch.nunique(axis=1) == 1).all()


def synthetic_oof_for_validation():
    return pd.DataFrame([
        dict(배치ID='b1', fold=1, target='Ni_회수율_pct', actual=80.0, prediction=81.0, signed_error_pp=1.0),
        dict(배치ID='b2', fold=1, target='Ni_회수율_pct', actual=82.0, prediction=83.0, signed_error_pp=1.0),
        dict(배치ID='b1', fold=1, target='Co_회수율_pct', actual=70.0, prediction=71.0, signed_error_pp=1.0),
        dict(배치ID='b2', fold=1, target='Co_회수율_pct', actual=72.0, prediction=73.0, signed_error_pp=1.0),
    ])


def write_official_csvs(tmp_path, monkeypatch, fold_scores_rows, comparison_rows):
    fold_scores_path, comparison_path = tmp_path / 'cv_fold_scores.csv', tmp_path / 'cv_comparison.csv'
    pd.DataFrame(fold_scores_rows).to_csv(fold_scores_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    monkeypatch.setattr(experiment, 'FOLD_SCORES_PATH', fold_scores_path)
    monkeypatch.setattr(experiment, 'COMPARISON_PATH', comparison_path)


def test_validate_oof_accepts_consistent_inputs(tmp_path, monkeypatch):
    # 두 배치 오차가 모두 1.0이므로 RMSE = sqrt(mean([1.0**2, 1.0**2])) = 1.0
    write_official_csvs(tmp_path, monkeypatch,
        fold_scores_rows=[dict(model='SVR', target=t, fold=1, val_rmse=1.0) for t in ['Ni_회수율_pct', 'Co_회수율_pct']],
        comparison_rows=[dict(model='SVR', target=t, val_rmse=1.0) for t in ['Ni_회수율_pct', 'Co_회수율_pct']])
    validate_oof(synthetic_oof_for_validation(), pd.DataFrame({'배치ID': ['b1', 'b2']}),
                 dict(Ni_회수율_pct=dict(model='SVR'), Co_회수율_pct=dict(model='SVR')))


def test_validate_oof_rejects_mismatched_fold_number_set(tmp_path, monkeypatch):
    # fold 개수는 동일(1개)하지만 번호가 1이 아니라 2로 밀려있고 값도 다름(99)
    write_official_csvs(tmp_path, monkeypatch,
        fold_scores_rows=[dict(model='SVR', target='Ni_회수율_pct', fold=2, val_rmse=99.0),
                          dict(model='SVR', target='Co_회수율_pct', fold=1, val_rmse=1.0)],
        comparison_rows=[dict(model='SVR', target=t, val_rmse=1.0) for t in ['Ni_회수율_pct', 'Co_회수율_pct']])
    with pytest.raises(ValueError, match='fold 번호'):
        validate_oof(synthetic_oof_for_validation(), pd.DataFrame({'배치ID': ['b1', 'b2']}),
                     dict(Ni_회수율_pct=dict(model='SVR'), Co_회수율_pct=dict(model='SVR')))


@pytest.mark.parametrize('comparison_rows', [
    [dict(model='SVR', target='Co_회수율_pct', val_rmse=1.0)],  # Ni 행 누락(0행)
    [dict(model='SVR', target='Ni_회수율_pct', val_rmse=1.0),
     dict(model='SVR', target='Ni_회수율_pct', val_rmse=9.0),
     dict(model='SVR', target='Co_회수율_pct', val_rmse=1.0)],  # Ni 행 중복(2행)
])
def test_validate_oof_rejects_missing_or_duplicated_comparison_row(tmp_path, monkeypatch, comparison_rows):
    write_official_csvs(tmp_path, monkeypatch,
        fold_scores_rows=[dict(model='SVR', target=t, fold=1, val_rmse=1.0) for t in ['Ni_회수율_pct', 'Co_회수율_pct']],
        comparison_rows=comparison_rows)
    with pytest.raises(ValueError, match='1행'):
        validate_oof(synthetic_oof_for_validation(), pd.DataFrame({'배치ID': ['b1', 'b2']}),
                     dict(Ni_회수율_pct=dict(model='SVR'), Co_회수율_pct=dict(model='SVR')))
