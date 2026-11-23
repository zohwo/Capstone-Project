"""B급의 Ni 목표 제약 아래 Co 최대화와 원료 투입량 민감도 비교함"""
import itertools
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from module1_preprocessing.transformation import prepare_split
from module2_prediction.artifacts import load_bundle
from module3_optimization.config import BOUNDS, CONTROL_COLUMNS, TARGETS, STEPS, SEEDS
from module3_optimization.genetic import genetic_search
from module3_optimization.optimization import candidate_frame, predict_candidates
from module3_optimization.validation import validate_feed, validate_controls, outcome


def co_objective(predictions, supported):
    predictions = np.asarray(predictions)
    valid = np.isfinite(predictions).all(axis=1) & ((predictions >= 0) & (predictions <= 100)).all(axis=1)
    valid &= np.asarray(supported, dtype=bool)
    ni, co = predictions[:, 0], predictions[:, 1]
    ni_target = TARGETS['Ni_회수율_pct']
    # Ni 목표 충족을 우선하고 충족 후보 안에서 Co 최대화함
    return np.where(valid, np.where(ni >= ni_target, 1 - co / 100, 2 + np.maximum(ni_target - ni, 0)), 1e9), valid


def maximize_co(bundles, feed, bounds=BOUNDS, enforce_training_ranges=False,
                known_controls=None, seeds=SEEDS, maxiter=60, grid_points=9):
    validate_feed(feed)
    if maxiter < 1:
        raise ValueError('maxiter는 1 이상이어야 함')
    bounds = np.asarray(bounds, dtype=float)
    references = {t: prepare_split(b['reference'], b['reference']) for t, b in bundles.items()}
    records, runs = [], []

    def evaluate(x, source):
        x = np.atleast_2d(x)
        pred = predict_candidates(bundles, feed, x)
        supported = np.ones(len(x), dtype=bool)
        if enforce_training_ranges:
            frame = candidate_frame(feed, x).assign(분석일자=pd.NaT, 원료등급=pd.NA)
            for t, b in bundles.items():
                prepared = prepare_split(b['reference'], frame)
                columns = b['features']
                supported &= ((prepared[columns] >= references[t][columns].min() - 1e-10)
                              & (prepared[columns] <= references[t][columns].max() + 1e-10)).all(axis=1).to_numpy()
        score, valid = co_objective(pred, supported)
        for controls, predictions, allowed in zip(x, pred, valid):
            records.append(dict(source=source, **dict(zip(CONTROL_COLUMNS, controls)),
                                **dict(zip(TARGETS, predictions)), valid=bool(allowed)))
        return score

    grid = np.array(list(itertools.product(*[np.linspace(lo, hi, grid_points) for lo, hi in bounds])))
    evaluate(grid, 'grid')
    if known_controls is not None:
        known = np.asarray(known_controls, dtype=float)
        if ((known >= bounds[:, 0]) & (known <= bounds[:, 1])).all():
            evaluate([known], 'known')
    for seed in seeds:
        result = differential_evolution(lambda x: evaluate(np.asarray(x).T, f'de_{seed}'), bounds,
            seed=seed, maxiter=maxiter, popsize=12, polish=False, vectorized=True, updating='deferred', tol=1e-7)
        runs.append(dict(algorithm='de', seed=seed, converged=bool(result.success), iterations=result.nit))
        runs.append(genetic_search(lambda x: evaluate(x, f'ga_{seed}'), bounds, seed, generations=maxiter))
    continuous = pd.DataFrame(records)
    rounded = np.round(continuous[CONTROL_COLUMNS].to_numpy() / STEPS) * STEPS
    rounded = np.clip(rounded, bounds[:, 0], bounds[:, 1])
    records.clear()
    for source, indices in continuous.groupby('source').groups.items():
        evaluate(np.unique(rounded[list(indices)], axis=0), source)
    candidates = pd.DataFrame(records)
    usable = candidates[candidates.valid].copy()
    if usable.empty:
        raise ValueError('허용 범위 내 유효 후보 미발견')
    usable['ni_deficit'] = np.maximum(TARGETS['Ni_회수율_pct'] - usable['Ni_회수율_pct'], 0)
    chosen = usable.sort_values(['ni_deficit', 'Co_회수율_pct'], ascending=[True, False]).iloc[0]
    controls = {c: float(chosen[c]) for c in CONTROL_COLUMNS}
    predictions = {t: float(chosen[t]) for t in TARGETS}
    validate_controls(list(controls.values()))
    valid, feasible, margins = outcome(predictions, TARGETS)
    checked = predict_candidates(bundles, feed, [list(controls.values())])[0]
    np.testing.assert_allclose(checked, list(predictions.values()), atol=1e-10)
    result = dict(feed=feed, controls=controls, predictions=predictions, margins=margins,
                  status='predicted_feasible' if feasible else 'target_not_found',
                  selected_source=chosen.source, runs=runs, training_ranges_enforced=enforce_training_ranges,
                  objective=f"Ni >= {TARGETS['Ni_회수율_pct']:g}% 제약 아래 Co 예측 회수율 최대화", bounds=bounds.tolist())
    return result, candidates


def main():
    root = Path('module3_optimization/reports')
    out = root / 'b_grade_capacity'
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root / 'representative_blends/B/recommendation.json').read_text())
    feed = previous['feed']
    models = Path('module2_prediction/reports/final')
    bundles = {t: load_bundle(models / f'{t}.joblib') for t in TARGETS}
    references = {t: prepare_split(b['reference'], b['reference']) for t, b in bundles.items()}
    bounds = np.asarray(BOUNDS, dtype=float).copy()
    for i, column in enumerate(CONTROL_COLUMNS):
        lo = max(bounds[i, 0], *(r[column].min() for r in references.values()))
        hi = min(bounds[i, 1], *(r[column].max() for r in references.values()))
        bounds[i] = [np.ceil(lo / STEPS[i]) * STEPS[i], np.floor(hi / STEPS[i]) * STEPS[i]]
    cases = [('dictionary_140', 140, BOUNDS, False)]
    cases += [(f'training_{ore}', ore, bounds, True) for ore in [81, 100, 120, 140, 160, 180, 199]]
    metadata = dict(feed=feed, original=previous, training_bounds=bounds.tolist(),
                    model_sha256={t: hashlib.sha256((models / f'{t}.joblib').read_bytes()).hexdigest() for t in TARGETS},
                    test_used=False, final_models_changed=False,
                    limitations=['학습 변수별 범위 안이어도 조합 전체의 학습 지원을 보장하지 않음',
                                 '예측상 목표 충족은 실제 실험 검증을 대신하지 않음',
                                 '원료량 감소는 배치당 처리량 감소이며 시간당 생산성·실제 비용은 평가하지 않음'])
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    rows = []
    for name, ore, search_bounds, restricted in cases:
        result, candidates = maximize_co(bundles, dict(feed, 원료투입량_kg=ore), bounds=search_bounds,
            enforce_training_ranges=restricted, known_controls=[previous['controls'][c] for c in CONTROL_COLUMNS])
        (out / f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        candidates.to_csv(out / f'{name}_candidates.csv', index=False)
        rows.append(dict(scenario=name, ore_kg=ore, training_ranges_enforced=restricted,
                         **result['controls'], **result['predictions'], status=result['status'],
                         ore_change_pct=(ore / 140 - 1) * 100,
                         predicted_Ni_kg=ore * feed['Ni_함량_pct'] / 100 * result['predictions']['Ni_회수율_pct'] / 100,
                         predicted_Co_kg=ore * feed['Co_함량_pct'] / 100 * result['predictions']['Co_회수율_pct'] / 100))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary.csv', index=False)
    subset = summary[summary.training_ranges_enforced]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, metal in zip(axes, ['Ni', 'Co']):
        recovery_column = f'{metal}_회수율_pct'
        target = TARGETS[recovery_column]
        ax.plot(subset.ore_kg, subset[recovery_column], 'o-')
        ax.axhline(target, color='red', linestyle='--', label=f'Target {target:g}%')
        ax.set(xlabel='Ore per batch (kg)', ylabel=f'Predicted {metal} recovery (%)', ylim=(0, 100))
        ax.legend()
    fig.suptitle(f"B representative feed | Maximize Co subject to Ni >= {TARGETS['Ni_회수율_pct']:g}%")
    fig.savefig(out / 'ore_sensitivity.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
