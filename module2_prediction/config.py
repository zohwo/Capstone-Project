"""Module 1의 공통 데이터 설정을 재사용하고 모델 탐색 후보 정의"""
from pathlib import Path

from module1_preprocessing.config import (
    BASE_FEATURES,
    COMPOSITION_COLUMNS,
    DERIVED_FEATURES,
    NORMAL_RANGES,
    PROCESS_COLUMNS,
    TARGET_COLUMNS,
)

PROCESS_COLUMN_RANGES = {column: NORMAL_RANGES[column] for column in PROCESS_COLUMNS}
FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES
SELECTABLE_FEATURE_COLUMNS = FEATURE_COLUMNS + ["원료등급"]

CLEANED_DATA_PATH = Path("data/processed/cleaned_features.csv")  # 품질 확인용
PRE_IMPUTE_DATA_PATH = Path("data/processed/pre_impute_data.csv")

_RF_SETTINGS = [
    # n_estimators, max_depth, min_samples_leaf
    (100, None, 1), (200, None, 1), (300, None, 1),
    (200, None, 2), (200, None, 4),
    (100, 10, 1), (200, 10, 2),
    (100, 5, 1), (200, 5, 1),
]

_BOOSTING_SETTINGS = [
    # n_estimators, max_depth, learning_rate, sampling
    (100, 2, 0.1, False), (200, 2, 0.1, False), (300, 2, 0.05, False),
    (100, 3, 0.05, False), (200, 3, 0.03, False), (300, 3, 0.03, False),
    (200, 3, 0.1, True), (200, 2, 0.1, True),
    (100, 4, 0.05, False), (200, 4, 0.03, False),
]


def _boosting_grid(*, lightgbm=False):
    """공통 후보를 생성하며 LightGBM의 행 샘플링 활성화 설정 추가"""
    candidates = []
    for n_estimators, max_depth, learning_rate, sampling in _BOOSTING_SETTINGS:
        params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        if sampling:
            params.update(subsample=0.8, colsample_bytree=0.8)
            if lightgbm:
                params['subsample_freq'] = 1
        candidates.append(params)
    return candidates


MODEL_HYPERPARAMETER_GRIDS = {
    "RandomForest": [
        dict(n_estimators=n, max_depth=depth, min_samples_leaf=leaf)
        for n, depth, leaf in _RF_SETTINGS
    ],
    "XGBoost": _boosting_grid(),
    "LightGBM": _boosting_grid(lightgbm=True),
    "ElasticNet": [
        dict(alpha=alpha, l1_ratio=ratio)
        for alpha in (0.01, 0.1, 1.0)
        for ratio in (0.1, 0.5, 0.9)
    ],
    "SVR": (
        [dict(C=c, gamma=gamma, epsilon=0.1)
         for gamma in ('scale', 'auto') for c in (1, 10, 100)]
        + [dict(C=10, gamma=gamma, epsilon=0.1) for gamma in (0.01, 0.1)]
        + [dict(C=10, gamma='scale', epsilon=epsilon) for epsilon in (0.5, 1.0)]
    ),
}
