"""개발 데이터 등급별 대표 조성으로 단독·혼합 시나리오 비교함"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS
from module3_optimization.config import TARGETS
from module3_optimization.pipeline import run
from module3_optimization.validation import validate_feed


def representative_feeds(data, ore_kg=140):
    development = data.loc[data['split'].isin(['train', 'validation'])]
    feeds = {}
    for grade in ['A', 'B', 'C']:
        group = development.loc[development['원료등급'] == grade, COMPOSITION_COLUMNS]
        if group.empty or group.notna().sum().eq(0).any():
            raise ValueError(f'{grade} 등급의 관측 조성 부족함')
        feed = dict(group.median(), 원료투입량_kg=ore_kg)
        validate_feed(feed)
        feeds[grade] = feed
    return feeds, development


def scenarios(feeds):
    cases = dict(feeds)
    for first, second in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        for numerator in [3, 5, 7]:
            ratio = numerator / 10
            feed = {c: ratio * feeds[first][c] + (1 - ratio) * feeds[second][c]
                    for c in COMPOSITION_COLUMNS}
            feed['원료투입량_kg'] = feeds[first]['원료투입량_kg']
            validate_feed(feed)
            cases[f'{first}{second}_{numerator}_{10-numerator}'] = feed
    return cases


def main():
    source = Path('data/processed/pre_impute_data.csv')
    out = Path('module3_optimization/reports/representative_blends')
    out.mkdir(parents=True, exist_ok=True)
    # 타깃은 읽지 않고 분할·등급·관측 조성만 사용함
    data = pd.read_csv(source, usecols=['split', '원료등급'] + COMPOSITION_COLUMNS)
    feeds, development = representative_feeds(data)
    metadata = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    development_rows=len(development), grade_counts=development['원료등급'].value_counts().to_dict(),
                    observed_counts=development.groupby('원료등급')[COMPOSITION_COLUMNS].count().to_dict('index'),
                    representative_feeds=feeds, total_ore_kg=140,
                    method='대치 전 개발 데이터에서 결측을 제외한 등급별 성분 중앙값',
                    limitations=['대표 조성은 실측 단일 배치가 아님', '목표 미발견은 불가능의 증명이 아님',
                                 '혼합 조성은 동일 질량 기준을 가정함', 'Test는 대표 조성 산출·최적화에 사용하지 않음'])
    (out / 'representatives.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    pd.DataFrame(feeds).T.to_csv(out / 'representative_compositions.csv', index_label='grade')
    rows = []
    for name, feed in scenarios(feeds).items():
        request = out / f'{name}_input.json'
        request.write_text(json.dumps(dict(description='개발 데이터 중앙값 기반 가상 대표 원료', feed=feed), ensure_ascii=False, indent=2))
        result = run(request, output_dir=out / name)
        rows.append(dict(scenario=name, status=result['status'], **result['controls'], **result['predictions'],
                         Ni_margin=result['margins_pct_point']['Ni_회수율_pct'],
                         Co_margin=result['margins_pct_point']['Co_회수율_pct'],
                         burden=result['burden'], selected_source=result['selected_source'],
                         outside_training_ranges=any(result['outside_training_feature_ranges'].values())))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout='constrained')
    for ax, metal in zip(axes, ['Ni', 'Co']):
        recovery_column = f'{metal}_회수율_pct'
        target = TARGETS[recovery_column]
        ax.bar(summary.scenario, summary[recovery_column])
        ax.axhline(target, color='red', linestyle='--', label=f'Target {target:g}%')
        ax.set(ylabel=f'{metal} recovery (%)', ylim=(0, 100))
        ax.legend(loc='lower right')
    axes[-1].set_xlabel('Feed scenario (mass ratio)')
    fig.suptitle('Predicted recovery at recommended conditions | Ore 140 kg')
    fig.savefig(out / 'blend_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
