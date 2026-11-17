import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from module1_preprocessing import pipeline
from module1_preprocessing.config import BASE_FEATURES, DERIVED_FEATURES, REFERENCE_EXCEPTION_RANGES, TARGET_COLUMNS
from module1_preprocessing.pipeline import run
from module1_preprocessing.validation import check_ranges
from module2_prediction.data_loading import load_modeling_inputs


def test_run_raises_on_conflicting_duplicate_batch(tmp_path):
    """같은 배치ID인데 값이 다른 행이 있으면, 분할 전에 바로 멈춰야 한다 —
    안 멈추면 같은 배치가 Train/Test에 걸쳐 들어가 데이터 누수가 생길 수 있다."""

    row = {
        "배치ID": "BP-2025-0001",
        "분석일자": "2025-01-01",
        "원료등급": "A",
        "Ni_함량_pct": 20.0,
        "Co_함량_pct": 8.0,
        "Li_함량_pct": 4.0,
        "Mn_함량_pct": 10.0,
        "Cu_불순물_pct": 1.0,
        "Fe_불순물_pct": 1.0,
        "Al_불순물_pct": 1.0,
        "원료투입량_kg": 140.0,
        "황산투입량_L": 900.0,
        "침출온도_C": 65.0,
        "침출시간_hr": 4.0,
        "Ni_회수율_pct": 85.0,
        "Co_회수율_pct": 85.0,
    }
    conflicting_row = dict(row, Ni_함량_pct=25.0)  # 같은 배치ID, Ni_함량_pct만 다름
    input_path = tmp_path / "raw.csv"
    pd.DataFrame([row, conflicting_row]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="충돌 중복"):
        run(input_path, tmp_path / "cleaned_features.csv")


def test_run_raises_clear_error_when_ore_mass_is_zero(tmp_path):
    """원료투입량_kg=0이면 acid_to_ore_ratio를 못 만드는데, 전체 컬럼 결측 검사가
    아니라 이 구체적인 원인을 짚어주는 에러가 먼저 나야 한다."""

    rng_rows = []
    for i in range(30):
        rng_rows.append({
            "배치ID": f"BP-2025-{i:04d}",
            "분석일자": "2025-01-01",
            "원료등급": ["A", "B", "C"][i % 3],
            "Ni_함량_pct": 20.0,
            "Co_함량_pct": 8.0,
            "Li_함량_pct": 4.0,
            "Mn_함량_pct": 10.0,
            "Cu_불순물_pct": 1.0,
            "Fe_불순물_pct": 1.0,
            "Al_불순물_pct": 1.0,
            "원료투입량_kg": 140.0,
            "황산투입량_L": 900.0,
            "침출온도_C": 65.0,
            "침출시간_hr": 4.0,
            "Ni_회수율_pct": 85.0,
            "Co_회수율_pct": 85.0,
        })
    rng_rows[0]["원료투입량_kg"] = 0.0

    input_path = tmp_path / "raw.csv"
    pd.DataFrame(rng_rows).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="원료투입량_kg"):
        run(input_path, tmp_path / "cleaned_features.csv")


def test_preserves_shared_features_despite_different_cv_winners(tmp_path, monkeypatch):
    from module1_preprocessing import pipeline
    from module1_preprocessing.config import BASE_FEATURES, DERIVED_FEATURES, TARGET_COLUMNS

    frame = pd.DataFrame({
        '배치ID': ['B1', 'B2', 'B3'],
        'split': ['train', 'validation', 'test'],
        **{target: [80.0, 85.0, 90.0] for target in TARGET_COLUMNS},
    })

    def fake_cv(pool, cases, targets, **kwargs):
        assert set(pool['split']) == {'train', 'validation'}
        return {(case, target): {'cv_mae': 1.0 if case == ('all_features' if target == TARGET_COLUMNS[0] else 'base') else 2.0}
                for case in cases for target in targets}

    monkeypatch.setattr(pipeline, 'run_cv_ablation_test', fake_cv)
    result = pipeline.select_and_save_features(frame, tmp_path / 'cleaned_features.csv')
    assert result['selection_policy'] == 'approved_shared_features'
    for target in TARGET_COLUMNS:
        expected_case = 'all_features' if target == TARGET_COLUMNS[0] else 'base'
        assert result['lowest_mae_cases'][target] == expected_case
        assert result['targets'][target]['features'] == BASE_FEATURES + DERIVED_FEATURES
        assert result['targets'][target]['case'] == 'base_plus_derived'
        assert result['targets'][target]['cv_metrics']['cv_mae'] == 2.0


def test_synthetic_pipeline_and_module2_handoff(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n = 120
    frame = pd.DataFrame({
        '배치ID': [f'BP-{i:04d}' for i in range(n)],
        '분석일자': ['03.01.2025', '01/08/25', '2025-01-12'] * 40,
        '원료등급': ['a급', ' B ', 'c'] * 40,
        'Ni_함량_pct': rng.uniform(10, 25, n),
        'Co_함량_pct': rng.uniform(3, 10, n),
        'Li_함량_pct': rng.uniform(2, 5, n),
        'Mn_함량_pct': rng.uniform(5, 12, n),
        'Cu_불순물_pct': rng.uniform(0.5, 3, n),
        'Fe_불순물_pct': rng.uniform(0.5, 3, n),
        'Al_불순물_pct': rng.uniform(0.5, 3, n),
        '원료투입량_kg': rng.uniform(100, 180, n),
        '침출온도_C': rng.uniform(50, 90, n),
        '침출시간_hr': rng.uniform(2, 7, n),
        **{t: rng.uniform(70, 95, n) for t in TARGET_COLUMNS},
    })
    frame['황산투입량_L'] = frame['원료투입량_kg'] * 6
    frame.loc[0, 'Ni_함량_pct'] = 6.5
    frame.loc[1, 'Ni_함량_pct'] = 999
    frame.loc[2, '황산투입량_L'] = np.nan
    frame.loc[3, '침출온도_C'] = np.nan
    frame = pd.concat([frame, frame.iloc[[4]]], ignore_index=True)
    source = tmp_path / 'synthetic.csv'
    frame.to_csv(source, index=False)
    result = run(source, tmp_path / 'cleaned_features.csv')
    assert len(result) == n
    assert result['배치ID'].is_unique
    assert result['split'].value_counts().to_dict() == {'train': 84, 'validation': 18, 'test': 18}
    assert np.isfinite(result[BASE_FEATURES + DERIVED_FEATURES]).all().all()
    assert result.loc[0, 'Ni_함량_pct'] == 6.5
    assert result.loc[0, 'Ni_함량_pct_dict_range_flag']
    assert result['is_duplicate_batch'].sum() == 1
    assert not result['is_conflicting_duplicate'].any()
    assert check_ranges(result, reference_exception_ranges=REFERENCE_EXCEPTION_RANGES) == {}
    for split in ('train', 'validation', 'test'):
        assert set(result.loc[result['split'].eq(split), '원료등급']) == {'A', 'B', 'C'}

    data, selection = load_modeling_inputs(tmp_path / 'pre_impute_data.csv')
    assert pd.isna(data.loc[1, 'Ni_함량_pct'])
    # 대치 전·후 CSV 모두 원래 결측과 이상치 처리 이력을 보존한다.
    cleaned = pd.read_csv(tmp_path / 'cleaned_features.csv')
    for saved in (data, cleaned):
        assert saved.loc[1, 'Ni_함량_pct_outlier_original_value'] == 999
        assert saved.loc[1, 'Ni_함량_pct_outlier_reason'] == 'outside_accepted_range'
        assert saved.loc[2, '황산투입량_L_outlier_reason'] == 'original_missing'
        assert saved.loc[0, 'Ni_함량_pct_outlier_reason'] == 'not_replaced'
    for target in TARGET_COLUMNS:
        assert selection['targets'][target]['features'] == BASE_FEATURES + DERIVED_FEATURES
        assert 'cv_rmse_std' in selection['targets'][target]['cv_metrics']
        np.testing.assert_allclose(result[target], frame.iloc[:n][target])

    original = pipeline.engineer_features
    def nonfinite_feature(data):
        out = original(data)
        out.loc[out.index[0], 'leach_intensity'] = np.inf
        return out
    monkeypatch.setattr(pipeline, 'engineer_features', nonfinite_feature)
    with pytest.raises(ValueError, match='무한대'):
        run(source, tmp_path / 'invalid/cleaned_features.csv')
    assert not (tmp_path / 'invalid/feature_selection.json').exists()



def test_evaluate_then_save_runs_cv_once(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        '배치ID': ['BP-001', 'BP-002', 'BP-003'],
        '원료등급': ['A', 'B', 'C'],
        'split': ['train', 'validation', 'test'],
        **{target: [80., 85., 90.] for target in TARGET_COLUMNS},
    })
    calls = []

    def fake_cv(pool, feature_sets, targets, **kwargs):
        calls.append(pool['배치ID'].tolist())
        return {(name, target): {'cv_mae': 1.0}
                for name in feature_sets for target in targets}

    monkeypatch.setattr(pipeline, 'run_cv_ablation_test', fake_cv)
    selection, results = pipeline.evaluate_feature_selection(frame)
    saved = pipeline.save_feature_selection(frame, selection, tmp_path / 'cleaned_features.csv')
    assert calls == [['BP-001', 'BP-002']]
    assert len(results) == 8
    assert saved == json.loads((tmp_path / 'feature_selection.json').read_text())
    assert saved['data_sha256'] == hashlib.sha256((tmp_path / 'pre_impute_data.csv').read_bytes()).hexdigest()

    changed = frame.copy()
    changed.loc[0, TARGET_COLUMNS[0]] += 1
    before = (tmp_path / 'pre_impute_data.csv').read_bytes()
    with pytest.raises(ValueError, match='입력 데이터가 변경됨'):
        pipeline.save_feature_selection(changed, selection, tmp_path / 'cleaned_features.csv')
    assert (tmp_path / 'pre_impute_data.csv').read_bytes() == before
    assert len(calls) == 1
