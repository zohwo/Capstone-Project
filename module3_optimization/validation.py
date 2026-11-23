"""추천 입력과 결과 검증함"""
import numpy as np
import pandas as pd
from module1_preprocessing.config import COMPOSITION_COLUMNS, NORMAL_RANGES, REFERENCE_EXCEPTION_RANGES
from module1_preprocessing.imputation import accepted_range_mask
from module3_optimization.config import CONTROL_COLUMNS, BOUNDS, TARGETS


def validate_feed(feed):
    required = COMPOSITION_COLUMNS + ['원료투입량_kg']
    if not set(required).issubset(feed):
        raise ValueError('조성 7종과 원료투입량 모두 필요함')
    for c in required:
        value = float(feed[c])
        if not accepted_range_mask(pd.Series([value]), NORMAL_RANGES[c], REFERENCE_EXCEPTION_RANGES.get(c, ())).iloc[0]:
            raise ValueError(f'{c}이 정상범위 또는 문헌 기반 보존 기준 밖임')
    if sum(float(feed[c]) for c in COMPOSITION_COLUMNS) > 100 + 1e-9:
        raise ValueError('동일 질량 기준의 성분 합계가 100% 초과함')


def validate_controls(values):
    x = np.asarray(values, dtype=float)
    valid_shape = x.ndim == 1 or (x.ndim == 2 and x.shape[0] >= 1)
    if not valid_shape or x.shape[-1] != len(CONTROL_COLUMNS):
        raise ValueError('운전조건 배열은 (3,) 또는 행이 1개 이상인 (N, 3) 형상이어야 함')
    bounds = np.asarray(BOUNDS)
    if not np.isfinite(x).all() or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError('운전조건이 프로젝트 탐색 범위 밖임')


def outcome(predictions, targets):
    if set(targets) != set(TARGETS):
        raise ValueError(f'목표는 {sorted(TARGETS)} 금속 키를 모두 가져야 함')
    if set(predictions) != set(targets):
        raise ValueError(f'예측값 키가 목표와 다름: {sorted(set(predictions) ^ set(targets))}')
    bad_targets = {t: v for t, v in targets.items() if not (np.isfinite(v) and 0 <= v <= 100)}
    if bad_targets:
        raise ValueError(f'목표값이 유한한 0~100 범위가 아님: {bad_targets}')
    valid = all(np.isfinite(v) and 0 <= v <= 100 for v in predictions.values())
    margins = {t: float(predictions[t] - targets[t]) for t in targets}
    feasible = valid and all(v >= 0 for v in margins.values())
    return valid, feasible, margins
