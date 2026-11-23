"""모델 로드부터 추천 조건 검증·저장까지 실행함"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from module2_prediction.artifacts import load_bundle
from module3_optimization.config import TARGETS, CONTROL_COLUMNS, BOUNDS
from module3_optimization.optimization import optimize, predict_candidates
from module3_optimization.validation import validate_controls


def run(input_path, model_dir='module2_prediction/reports/final', output_dir='module3_optimization/reports/recommendation', **options):
    request = json.loads(Path(input_path).read_text(encoding='utf-8'))
    bundles = {t: load_bundle(Path(model_dir) / f'{t}.joblib') for t in TARGETS}
    for t, b in bundles.items():
        if b['target'] != t:
            raise ValueError('모델 타깃 정보 불일치함')
    result, candidates = optimize(bundles, request['feed'], targets=request.get('targets'),
                                  weights=request.get('weights'), **options)
    result['input_description'] = request.get('description', '')
    result['model_sha256'] = {t: hashlib.sha256((Path(model_dir) / f'{t}.joblib').read_bytes()).hexdigest() for t in TARGETS}
    if 'current_controls' in request:
        current = [request['current_controls'][c] for c in CONTROL_COLUMNS]
        validate_controls(current)
        predicted = predict_candidates(bundles, request['feed'], [current])[0]
        if np.isfinite(predicted).all() and ((predicted >= 0) & (predicted <= 100)).all():
            result['current'] = dict(controls=request['current_controls'], predictions=dict(zip(TARGETS, predicted.tolist())))
            result['predicted_change_pct_point'] = {t: result['predictions'][t] - float(p) for t, p in zip(TARGETS, predicted)}
        else:
            result['current'] = dict(controls=request['current_controls'], predictions=None,
                comparison_excluded_reason='기존 운전조건의 Ni·Co 예측값 중 하나 이상이 유한한 0~100% 범위 조건을 충족하지 않음')
    x = [result['controls'][c] for c in CONTROL_COLUMNS]
    checked = predict_candidates(bundles, request['feed'], [x])[0]
    if not np.allclose(checked, [result['predictions'][t] for t in TARGETS], atol=1e-10):
        raise ValueError('추천 조건 재예측 결과 불일치함')
    result['recommendation_reprediction_verified'] = True
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / 'candidates.csv', index=False)
    comparison = candidates[candidates.valid].sort_values(['feasible', 'deficit', 'burden'], ascending=[False, True, True]).groupby('source', sort=False).head(1)
    comparison.to_csv(out / 'search_comparison.csv', index=False)
    (out / 'recommendation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_surface(bundles, request['feed'], result, out)
    return result


def plot_surface(bundles, feed, result, output_dir):
    acid = np.linspace(*BOUNDS[0], 31)
    temp = np.linspace(*BOUNDS[1], 31)
    aa, tt = np.meshgrid(acid, temp)
    controls = np.column_stack([aa.ravel(), tt.ravel(), np.full(aa.size, result['controls']['침출시간_hr'])])
    values = predict_candidates(bundles, feed, controls)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i, (ax, target) in enumerate(zip(axes, TARGETS)):
        z = values[:, i].reshape(aa.shape)
        im = ax.pcolormesh(aa, tt, z, shading='auto', vmin=0, vmax=100, cmap='viridis')
        level = result['targets'][target]
        if z.min() < level < z.max():
            ax.contour(aa, tt, z, levels=[level], colors='white')
        ax.scatter(result['controls']['황산투입량_L'], result['controls']['침출온도_C'], marker='*', color='red', s=90)
        ax.set(xlabel='Acid solution (L)', ylabel='Temperature (C)', title=f'{target.split("_")[0]} predicted recovery (%)')
    fig.colorbar(im, ax=axes, label='Predicted recovery (%)')
    fig.suptitle(f'Time fixed at {result["controls"]["침출시간_hr"]:.1f} h')
    fig.savefig(Path(output_dir) / 'recovery_surface.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-dir', default='module2_prediction/reports/final')
    parser.add_argument('--output-dir', default='module3_optimization/reports/recommendation')
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.model_dir, args.output_dir), ensure_ascii=False, indent=2))
