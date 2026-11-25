"""고정 원료에서 격자 탐색·DE·GA로 운전조건을 탐색하고 설정 단위로 반올림한 후보를 재평가함."""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from module1_preprocessing.transformation import prepare_split
from module2_prediction.artifacts import predict_bundle
from module3_optimization.config import CONTROL_COLUMNS, BOUNDS, TARGETS, STEPS, WEIGHTS, SEEDS
from module3_optimization.validation import validate_feed, validate_controls, outcome
from module3_optimization.genetic import genetic_search


def candidate_frame(feed, controls):
    x = np.atleast_2d(np.asarray(controls, dtype=float))
    validate_controls(x)
    frame = pd.DataFrame([feed] * len(x))
    for i, c in enumerate(CONTROL_COLUMNS):
        frame[c] = x[:, i]
    return frame


def predict_candidates(bundles, feed, controls):
    frame = candidate_frame(feed, controls)
    return np.column_stack([predict_bundle(bundles[t], frame) for t in TARGETS])


def optimize(bundles, feed, targets=None, weights=None, seeds=SEEDS, maxiter=60, grid_points=9):
    validate_feed(feed)
    targets = dict(TARGETS if targets is None else targets)
    if set(targets) != set(TARGETS) or not all(np.isfinite(v) and 0 <= v <= 100 for v in targets.values()):
        raise ValueError('Ni·Co 목표는 유한한 0~100% 값이어야 함')
    weights = np.asarray(WEIGHTS if weights is None else weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('운전 부담 가중치 3개는 0 이상이며 합계가 양수여야 함')
    if not seeds or maxiter < 1 or grid_points < 2:
        raise ValueError('탐색 seed·반복 수·격자 수 확인 필요함')
    weights = weights / weights.sum()
    bounds = np.asarray(BOUNDS, dtype=float)
    goals = np.array([targets[t] for t in TARGETS])
    records, runs = [], []

    def evaluate(x, source):
        x = np.atleast_2d(x)
        predictions = predict_candidates(bundles, feed, x)
        burden = ((x - bounds[:, 0]) / np.diff(bounds, axis=1).ravel()) @ weights
        valid = np.isfinite(predictions).all(axis=1) & ((predictions >= 0) & (predictions <= 100)).all(axis=1)
        deficit = np.maximum(goals - predictions, 0).sum(axis=1)
        feasible = valid & (deficit == 0)
        scores = np.where(valid, np.where(feasible, burden, 1 + deficit + 1e-6 * burden), 1e9)
        for a, p, b, d, v, f in zip(x, predictions, burden, deficit, valid, feasible):
            records.append(dict(source=source, **dict(zip(CONTROL_COLUMNS, a)),
                **dict(zip(TARGETS, p)), burden=float(b), deficit=float(d), valid=bool(v), feasible=bool(f)))
        return scores

    grid = np.array(list(itertools.product(*[np.linspace(lo, hi, grid_points) for lo, hi in BOUNDS])))
    evaluate(grid, 'grid')
    for seed in seeds:
        before = len(records)
        def objective(x):
            return evaluate(np.asarray(x).T, f'de_{seed}')
        result = differential_evolution(objective, BOUNDS, seed=int(seed), maxiter=maxiter,
            popsize=12, polish=False, vectorized=True, updating='deferred', tol=1e-6)
        runs.append(dict(algorithm='de', seed=int(seed), converged=bool(result.success), message=str(result.message), iterations=int(result.nit), evaluations=len(records) - before))
    for seed in seeds:
        runs.append(genetic_search(lambda x: evaluate(x, f'ga_{seed}'), BOUNDS,
                                   seed=int(seed), generations=maxiter))
    continuous = pd.DataFrame(records)
    rounded = np.round(continuous[CONTROL_COLUMNS].to_numpy() / STEPS) * STEPS
    rounded = np.clip(rounded, bounds[:, 0], bounds[:, 1])
    records.clear()
    for source, indexes in continuous.groupby('source').groups.items():
        evaluate(np.unique(rounded[list(indexes)], axis=0), source)
    candidates = pd.DataFrame(records)
    usable = candidates[candidates.valid]
    if usable.empty:
        raise ValueError('유효한 회수율 예측 후보를 얻지 못함')
    ordered = usable.sort_values(['feasible', 'deficit', 'burden'], ascending=[False, True, True])
    chosen = ordered.iloc[0]
    predictions = {t: float(chosen[t]) for t in TARGETS}
    valid, feasible, margins = outcome(predictions, targets)
    controls = {c: float(chosen[c]) for c in CONTROL_COLUMNS}
    validate_controls(list(controls.values()))
    frame = candidate_frame(feed, [list(controls.values())])
    coverage = {}
    for t, bundle in bundles.items():
        reference = prepare_split(bundle['reference'], bundle['reference'])
        row = prepare_split(bundle['reference'], frame.assign(분석일자=pd.NaT, 원료등급=pd.NA))
        columns = [c for c in bundle['features'] if c != '원료등급']
        coverage[t] = [c for c in columns if row[c].iloc[0] < reference[c].min() or row[c].iloc[0] > reference[c].max()]
    summary = dict(status='predicted_feasible' if feasible else 'target_not_found',
        feed=feed, targets=targets, controls=controls, predictions=predictions, margins_pct_point=margins,
        burden=float(chosen.burden), selected_source=chosen.source, weights=weights.tolist(),
        boundary_controls=[c for c, (lo, hi) in zip(CONTROL_COLUMNS, BOUNDS) if np.isclose(controls[c], lo) or np.isclose(controls[c], hi)],
        outside_training_feature_ranges=coverage, runs=runs,
        rounding_steps=dict(zip(CONTROL_COLUMNS, STEPS)), grid_points=grid_points,
        limitations=['회수율은 모델 예측이며 실제 달성을 보장하지 않음',
                    '운전 부담은 정규화 가중합이며 실제 비용이 아님',
                    '범위 내 입력이어도 학습 데이터의 조합 분포를 벗어날 수 있음',
                    '미발견은 탐색 예산 내 결과이며 목표 달성 불가능의 증명이 아님'])
    return summary, candidates
