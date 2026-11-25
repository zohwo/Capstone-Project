import json

import pandas as pd
import pytest

from module1_preprocessing.config import COMPOSITION_COLUMNS
from module3_optimization.config import CONTROL_COLUMNS
from module3_optimization.scripts import test_batch_examples as experiment
from module3_optimization.scripts.test_batch_examples import select_batches


def synthetic_rows():
    base = dict(split='test', **{c: 10.0 for c in COMPOSITION_COLUMNS},
                **{c: 50.0 for c in CONTROL_COLUMNS}, 원료투입량_kg=140.0,
                Ni_회수율_pct=90.0, Co_회수율_pct=80.0)
    return pd.DataFrame([
        dict(base, 배치ID='B1', 원료등급='A'),                            # 완전 — 선택돼야 함
        dict(base, 배치ID='B2', 원료등급='B', Co_회수율_pct=None),          # Co 실측 결측
        dict(base, 배치ID='B3', 원료등급='B', Ni_회수율_pct=None),          # Ni 실측 결측
        dict(base, 배치ID='B4', 원료등급='B', 황산투입량_L=None),            # 운전조건 결측
        dict(base, 배치ID='B5', 원료등급='B'),                            # 완전, B등급 — 그룹화 검증용
        dict(base, 배치ID='B6_train', 원료등급='A', split='train'),
        dict(base, 배치ID='B7', 원료등급=None),                           # 완전, 등급 결측 — '미기록' 그룹화 검증용
    ])


def test_select_batches_requires_actual_recovery_completeness():
    """Ni·Co 실측 회수율 중 하나라도 결측이면 조성·운전조건이 완전해도 제외돼야 함"""
    selected = select_batches(synthetic_rows())
    assert set(selected['배치ID']) == {'B1', 'B5', 'B7'}


def test_select_batches_raises_when_no_complete_test_batch():
    data = synthetic_rows()
    data = data[~data['배치ID'].isin(['B1', 'B5', 'B7'])]
    with pytest.raises(ValueError, match='없음'):
        select_batches(data)


def fake_run(request_path, output_dir):
    request = json.loads(request_path.read_text(encoding='utf-8'))
    return dict(status='predicted_feasible',
                controls=dict(황산투입량_L=1000.0, 침출온도_C=90.0, 침출시간_hr=5.0),
                predictions=dict(Ni_회수율_pct=96.0, Co_회수율_pct=91.0),
                current=dict(controls=request['current_controls'],
                             predictions=dict(Ni_회수율_pct=89.0, Co_회수율_pct=81.0)))


def test_main_writes_summary_and_groups_metadata_by_grade(tmp_path, monkeypatch):
    source = tmp_path / 'pre_impute_data.csv'
    synthetic_rows().to_csv(source, index=False)
    out_dir = tmp_path / 'out'
    monkeypatch.setattr(experiment, 'SOURCE', source)
    monkeypatch.setattr(experiment, 'OUTPUT_DIR', out_dir)
    monkeypatch.setattr(experiment, 'run', fake_run)

    experiment.main()

    summary = pd.read_csv(out_dir / 'summary.csv').set_index('batch_id')
    assert set(summary.index) == {'B1', 'B5', 'B7'}
    assert summary.loc['B1', 'grade'] == 'A'
    assert summary.loc['B5', 'grade'] == 'B'
    assert pd.isna(summary.loc['B7', 'grade'])
    # 기존 조건 실측값 / 기존 조건 모델 예측값 / 추천 조건 모델 예측값을 각각 확인함
    assert summary.loc['B1', 'actual_Ni_회수율_pct'] == 90.0
    assert summary.loc['B1', 'model_predicted_at_actual_control_Ni_회수율_pct'] == 89.0
    assert summary.loc['B1', 'predicted_Ni_회수율_pct'] == 96.0

    metadata = json.loads((out_dir / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['selected_batch_count'] == 3
    assert metadata['selected_batches'] == {'A': ['B1'], 'B': ['B5'], '미기록': ['B7']}
