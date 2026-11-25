"""공식 CV를 재현해 개발 데이터 357개 배치의 Ni·Co OOF 오차를 배치 단위로 짝지음"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from module1_preprocessing.transformation import prepare_split
from module2_prediction.config import PRE_IMPUTE_DATA_PATH, TARGET_COLUMNS
from module2_prediction.data_loading import load_modeling_inputs, split_by_column
from module2_prediction.modeling import MODEL_FACTORIES, fit_selected_model

RESULTS_PATH = Path('module2_prediction/reports/final/results.json')
FOLD_SCORES_PATH = Path('module2_prediction/reports/final/cv_fold_scores.csv')
COMPARISON_PATH = Path('module2_prediction/reports/final/cv_comparison.csv')
OUTPUT_DIR = Path('module2_prediction/reports/oof_margin')


def generate_oof(cv_pool, cv_config, results):
    """fold 분할을 한 번만 만들어 Ni·Co에 공통으로 재사용, 각 fold 학습행만으로 대치·표준화 적합"""
    cv = StratifiedKFold(n_splits=cv_config['n_splits'], shuffle=True, random_state=cv_config['random_state'])
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(cv_pool, cv_pool[cv_config['stratify_column']]), start=1):
        train_raw, val_raw = cv_pool.iloc[train_idx], cv_pool.iloc[val_idx]
        train_prepared = prepare_split(train_raw, train_raw)
        val_prepared = prepare_split(train_raw, val_raw)
        for target in TARGET_COLUMNS:
            info = results[target]
            columns = info['features']
            model = fit_selected_model(MODEL_FACTORIES[info['model']](**info['params']), train_prepared, columns, target)
            predictions = model.predict(val_prepared[columns])
            for batch_id, actual, prediction in zip(val_prepared['배치ID'], val_prepared[target], predictions):
                rows.append(dict(배치ID=batch_id, fold=fold, target=target,
                                  actual=float(actual), prediction=float(prediction),
                                  signed_error_pp=float(prediction - actual)))
    return pd.DataFrame(rows)


def validate_oof(oof, cv_pool, results):
    """저장 전 배치 귀속·fold 일치·공식 CV 지표 재현을 확인, 어긋나면 ValueError"""
    pool_ids = set(cv_pool['배치ID'])
    for target in TARGET_COLUMNS:
        subset = oof[oof.target == target]
        if len(subset) != len(cv_pool):
            raise ValueError(f'{target} OOF 행 수가 개발 배치 수와 다름: {len(subset)} != {len(cv_pool)}')
        ids = subset['배치ID']
        if ids.duplicated().any():
            raise ValueError(f'{target} OOF에 배치ID 중복 있음')
        if set(ids) != pool_ids:
            raise ValueError(f'{target} OOF 배치ID 집합이 개발 풀과 다름')
        if not np.isfinite(subset[['actual', 'prediction', 'signed_error_pp']].to_numpy()).all():
            raise ValueError(f'{target} OOF에 비유한 실측·예측·오차 값 있음')

    fold_by_batch = oof.pivot(index='배치ID', columns='target', values='fold')
    mismatched = fold_by_batch[fold_by_batch.nunique(axis=1) > 1]
    if not mismatched.empty:
        raise ValueError(f'배치별 Ni·Co fold 번호 불일치: {mismatched.index.tolist()}')

    fold_scores = pd.read_csv(FOLD_SCORES_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    for target in TARGET_COLUMNS:
        model_name = results[target]['model']
        recomputed = oof[oof.target == target].groupby('fold')['signed_error_pp'].apply(
            lambda e: float(np.sqrt(np.mean(np.square(e.to_numpy())))))
        official_folds = fold_scores[(fold_scores.model == model_name) & (fold_scores.target == target)].set_index('fold')['val_rmse']
        if len(official_folds) != len(recomputed):
            raise ValueError(f'{target}의 공식 fold RMSE 개수가 재현된 fold 수와 다름')
        if set(official_folds.index) != set(recomputed.index):
            raise ValueError(f'{target}의 fold 번호 집합이 서로 다름: {sorted(recomputed.index)} != {sorted(official_folds.index)}')
        if not np.isfinite(recomputed.to_numpy()).all() or not np.isfinite(official_folds.to_numpy()).all():
            raise ValueError(f'{target}의 fold별 RMSE에 비유한 값 있음')
        diff = (recomputed.sort_index() - official_folds.sort_index()).abs()
        if (diff > 1e-9).any():
            raise ValueError(f'{target}의 fold별 RMSE가 cv_fold_scores.csv와 불일치: {diff.to_dict()}')
        official_mean_rows = comparison.loc[(comparison.model == model_name) & (comparison.target == target), 'val_rmse']
        if len(official_mean_rows) != 1:
            raise ValueError(
                f'{target}/{model_name}의 공식 평균 RMSE는 1행이어야 함: '
                f'{len(official_mean_rows)}행'
            )
        if not np.isfinite(official_mean_rows.to_numpy()).all():
            raise ValueError(f'{target}의 공식 평균 RMSE에 비유한 값 있음')
        mean_rmse = float(recomputed.mean())
        official_mean = float(official_mean_rows.iloc[0])
        if abs(mean_rmse - official_mean) > 1e-9:
            raise ValueError(f'{target}의 fold 평균 RMSE가 cv_comparison.csv와 불일치: {mean_rmse} != {official_mean}')


def main():
    df, selection = load_modeling_inputs(PRE_IMPUTE_DATA_PATH)
    train_df, val_df, _ = split_by_column(df)
    cv_pool = pd.concat([train_df, val_df], ignore_index=True)
    results = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))

    oof = generate_oof(cv_pool, selection['cv'], results)
    validate_oof(oof, cv_pool, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUTPUT_DIR / 'oof_errors.csv', index=False)
    metadata = dict(
        development_rows=len(cv_pool), oof_rows=len(oof), targets=results, cv=selection['cv'],
        source_file=str(PRE_IMPUTE_DATA_PATH), source_sha256=hashlib.sha256(Path(PRE_IMPUTE_DATA_PATH).read_bytes()).hexdigest(),
        row_order='기존 Module 2와 동일한 Train 다음 Validation 순서',
        fold_assignment='Ni·Co가 동일한 StratifiedKFold 분할을 공유함',
        validated_against=dict(fold_scores=str(FOLD_SCORES_PATH), comparison=str(COMPARISON_PATH), tolerance=1e-9),
        test_used=False, final_models_changed=False,
        scope='각 배치의 OOF 예측은 해당 배치를 학습에서 제외한 모델로 생성했다. 모델과 하이퍼파라미터 선택에는 같은 개발 데이터를 사용했으므로, '
              '이 결과는 추천 여유폭을 탐색하는 데 사용한다. 실제 추천 배치의 Ni·Co 동시 목표 달성률은 별도로 검증해야 한다.')
    (OUTPUT_DIR / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'OOF rows: {len(oof)}', flush=True)
    print(oof.groupby('target')['signed_error_pp'].describe().to_string(), flush=True)


if __name__ == '__main__':
    main()
