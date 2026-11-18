"""개발 데이터에서 파생변수 6종의 단독 추가 및 전체 조합 비교

실행: python -m module1_preprocessing.scripts.select_derived_subset
Test 평가와 최종 설정 변경 없이 비교 결과만 저장
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from module1_preprocessing.config import BASE_FEATURES, TARGET_COLUMNS
from module1_preprocessing.transformation import prepare_split, validate_model_features, validate_targets

CANDIDATES = ['acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct',
              'feed_impurity_g_per_l', 'acid_per_target_L_kg', 'acid_time']

CURRENT_FOUR = ['leach_intensity', 'feed_impurity_g_per_l', 'acid_per_target_L_kg', 'acid_time']


def run(seeds=(42, 17, 83), tolerance=0.02):
    root = Path(__file__).resolve().parents[2]
    path = root / 'data/processed/pre_impute_data.csv'
    data = pd.read_csv(path)
    pool = pd.concat([data.loc[data['split'] == split] for split in ('train', 'validation')],
                      ignore_index=True)

    if pool['배치ID'].isna().any() or pool['배치ID'].duplicated().any():
        raise ValueError('개발 데이터 배치ID 누락 또는 중복 있음')
    validate_targets(pool, TARGET_COLUMNS)

    subsets = [list(items) for k in range(7) for items in itertools.combinations(CANDIDATES, k)]

    records = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(pool, pool['원료등급']), 1):
            train, val = pool.iloc[tr], pool.iloc[va]
            train_x, val_x = prepare_split(train, train), prepare_split(train, val)

            for frame in (train_x, val_x):
                validate_model_features(frame, BASE_FEATURES + CANDIDATES)

            for grade in (False, True):
                for case, subset in enumerate(subsets):
                    columns = BASE_FEATURES + subset + (['원료등급'] if grade else [])

                    X_train = pd.get_dummies(train_x[columns])

                    X_val = pd.get_dummies(val_x[columns]).reindex(
                        columns=X_train.columns, fill_value=0,
                    )

                    for target in TARGET_COLUMNS:
                        model = LGBMRegressor(random_state=42, verbose=-1, n_jobs=1)

                        model.fit(X_train, train[target])

                        mae = mean_absolute_error(val[target], model.predict(X_val))

                        records.append(dict(
                            seed=seed, fold=fold, grade=grade, case=case,
                            target=target, mae=mae,
                        ))
            print(f'seed {seed}, fold {fold}: 등급 제외·포함 각 64개 조합 완료함', flush=True)

    scores = pd.DataFrame(records)

    keys = ['target', 'grade', 'case']

    seed_scores = scores.groupby(keys + ['seed'], as_index=False)['mae'].mean()

    summary = seed_scores.groupby(keys)['mae'].agg(['mean', 'std', 'min', 'max']).reset_index()

    summary['count'] = summary['case'].map(lambda i: len(subsets[i]))

    summary['features'] = summary['case'].map(lambda i: '|'.join(subsets[i]))

    baseline = summary.loc[summary['case'] == 0, ['target', 'grade', 'mean']].rename(
        columns={'mean': 'baseline_mae'},
    )

    summary = summary.merge(baseline, on=['target', 'grade'])

    summary['delta_vs_base'] = summary['mean'] - summary['baseline_mae']

    output = root / 'module1_preprocessing/reports/derived_subset_selection'

    output.mkdir(parents=True, exist_ok=True)

    scores.to_csv(output / 'fold_scores.csv', index=False)

    seed_scores.to_csv(output / 'seed_scores.csv', index=False)

    summary.sort_values(keys[:2] + ['mean']).to_csv(output / 'subset_summary.csv', index=False)

    summary.loc[summary['count'] <= 1].to_csv(output / 'single_addition_summary.csv', index=False)

    seed_scores.loc[seed_scores['seed'] == 42].merge(
        summary[keys + ['count', 'features']], on=keys,
    ).to_csv(output / 'seed42_summary.csv', index=False)

    recommendations = []
    for (target, grade), part in summary.groupby(['target', 'grade']):
        best = part.sort_values(['mean', 'count']).iloc[0]

        simple = part.loc[part['mean'] <= best['mean'] + tolerance].sort_values(
            ['count', 'mean'],
        ).iloc[0]

        recommendations.append(dict(
            target=target, grade=bool(grade),
            best_features=subsets[int(best['case'])], best_mae=float(best['mean']),
            simpler_features=subsets[int(simple['case'])], simpler_mae=float(simple['mean']),
        ))

    common = summary.groupby(['grade', 'case', 'count', 'features'], as_index=False)['mean'].mean()

    common = common.rename(columns={'mean': 'mean_target_mae'}).sort_values('mean_target_mae')

    common.to_csv(output / 'common_subset_summary.csv', index=False)

    report = dict(
        development_rows=len(pool), test_used=False, seeds=list(seeds), folds=5,
        model='LightGBM defaults, random_state=42',
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        candidates=CANDIDATES, current_four=CURRENT_FOUR, combinations_per_grade=64,
        grade_conditions=[False, True], practical_tolerance_pct_point=tolerance,
        recommendations=recommendations,
        limitations=[
            '반복 CV 점수는 선택용이며 독립적인 일반화 성능이 아님',
            'fold와 반복 시드는 독립 표본이 아니므로 표준편차를 유의성 근거로 사용하지 않음',
            '허용 차이는 실용적인 참고 기준이며 통계적 동등성 기준이 아님',
        ],
    )

    (output / 'recommendation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tolerance', type=float, default=0.02)
    args = parser.parse_args()

    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error('tolerance는 0 이상의 유한한 값이어야 함')

    run(tolerance=args.tolerance)
