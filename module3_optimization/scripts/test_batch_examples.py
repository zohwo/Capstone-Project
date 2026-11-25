"""Test 배치의 실측 조성·운전조건을 데모 입력으로 사용함 — 설정 변경·모델 재선택에는 쓰지 않음"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS
from module3_optimization.config import CONTROL_COLUMNS
from module3_optimization.pipeline import run

SOURCE = Path('data/processed/pre_impute_data.csv')
OUTPUT_DIR = Path('module3_optimization/reports/test_batches')


def select_batches(data):
    """필수값이 모두 있는 Test 배치를 고름, 실측 회수율의 크기와 목표 달성 여부는 고려하지 않음"""
    required = COMPOSITION_COLUMNS + CONTROL_COLUMNS + ['원료투입량_kg', 'Ni_회수율_pct', 'Co_회수율_pct']
    test = data.loc[data['split'] == 'test'].copy()
    complete = test[required].notna().all(axis=1)
    selected = test.loc[complete].sort_values('배치ID')
    if selected.empty:
        raise ValueError('결측 없는 Test 배치가 없음')
    return selected


def main():
    data = pd.read_csv(SOURCE)
    selected = select_batches(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, batch in selected.iterrows():
        batch_id, grade = batch['배치ID'], batch['원료등급']
        feed = {c: float(batch[c]) for c in COMPOSITION_COLUMNS}
        feed['원료투입량_kg'] = float(batch['원료투입량_kg'])
        current_controls = {c: float(batch[c]) for c in CONTROL_COLUMNS}
        grade_label = '등급 미기록' if pd.isna(grade) else f'{grade}등급'
        request = dict(description=f'Test 배치 {batch_id}({grade_label})의 실측 조성·운전조건 — 데모 입력',
                       feed=feed, current_controls=current_controls)
        request_path = OUTPUT_DIR / f'{batch_id}_input.json'
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
        result = run(request_path, output_dir=OUTPUT_DIR / batch_id)
        row = dict(grade=grade, batch_id=batch_id, status=result['status'],
                  actual_Ni_회수율_pct=float(batch['Ni_회수율_pct']), actual_Co_회수율_pct=float(batch['Co_회수율_pct']),
                  **{f'actual_control_{c}': v for c, v in current_controls.items()},
                  **{f'recommended_{c}': v for c, v in result['controls'].items()},
                  **{f'predicted_{t}': v for t, v in result['predictions'].items()})
        if result['current']['predictions'] is not None:
            row.update({f'model_predicted_at_actual_control_{t}': v for t, v in result['current']['predictions'].items()})
        rows.append(row)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False)
    metadata = dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        selection_rule=('실측 조성·원료투입량·운전조건·Ni/Co 회수율이 모두 있는 Test 배치를 선정함. '
                        '실측 회수율의 크기와 목표 달성 여부는 고려하지 않음'),
        test_used_as=('실측 조성과 원료투입량은 추천 요청의 원료 입력으로 사용하고, '
                      '실측 운전조건은 기존 조건의 모델 예측에 사용함. '
                      '실측 회수율은 배치 선별 시 결측 여부 확인과 결과표 기록에 사용함'),
        test_not_used_for='Module 2 모델·피처 재선택, Module 3 설정(가중치·여유폭·알고리즘) 조정에는 사용하지 않음',
        total_test_batches=int((data['split'] == 'test').sum()),
        selected_batch_count=len(selected),
        selected_batches={grade: group['배치ID'].tolist()
                         for grade, group in selected.groupby(selected['원료등급'].fillna('미기록'))},
        limitations=['실측 운전조건 대비 추천 조건의 예측상 개선폭이며 실제 재실험 결과가 아님',
                     'Module 2 모델은 이 배치들을 포함해 선택된 적 없음(Test는 평가 전용으로 유지됨)',
                     '결측 있는 Test 배치는 제외되어 전체 Test를 대표하지 않을 수 있음'])
    (OUTPUT_DIR / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
