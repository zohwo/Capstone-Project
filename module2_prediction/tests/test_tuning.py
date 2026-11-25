import numpy as np
import pandas as pd
import pytest

from module2_prediction.config import MODEL_HYPERPARAMETER_GRIDS, TARGET_COLUMNS
from module2_prediction.tuning import tune_models_cv


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "원료등급": np.tile(["A", "B", "C"], n // 3),
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "acid_to_ore_ratio": rng.normal(7, 0.5, n),
        "leach_intensity": rng.normal(260, 40, n),
        "total_impurity_pct": rng.normal(6, 1, n),
        "feed_impurity_g_per_l": rng.normal(10.0, 3.0, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["원료등급"] = ["A", "B", "C"] * (n // 3)
    df["원료투입량_kg"] = 140.0
    df["황산투입량_L"] = 980.0
    df["분석일자"] = pd.Timestamp("2025-01-01")
    return df


def test_cv_selects_minimum_rmse_with_nonfinite_r2(monkeypatch):
    from sklearn.dummy import DummyRegressor
    from module2_prediction import tuning

    df = _make_fixture()
    target = TARGET_COLUMNS[0]
    df[target] = 70.0
    monkeypatch.setattr(tuning, 'MODEL_FACTORIES', {
        'Constant': lambda **params: DummyRegressor(strategy='constant', **params),
    })
    monkeypatch.setattr(tuning, 'MODEL_HYPERPARAMETER_GRIDS', {
        'Constant': [{'constant': 68.0}, {'constant': 69.0}],
    })
    with pytest.warns(RuntimeWarning, match='divide by zero'):
        best = tune_models_cv(df, n_splits=3, target_columns=[target])
    entry = best[('Constant', target)]
    assert entry['params'] == {'constant': 69.0}
    assert entry['metrics']['val_rmse'] == pytest.approx(1.0)
    assert entry['metrics']['val_mae'] == pytest.approx(1.0)
    assert np.isneginf(entry['metrics']['val_r2'])
    assert len(entry['fold_metrics']) == 3


def test_tune_models_cv_returns_best_params_per_model_and_target():
    df = _make_fixture()

    best = tune_models_cv(df, n_splits=5)

    assert set(best.keys()) == {
        (model_name, target)
        for model_name in MODEL_HYPERPARAMETER_GRIDS
        for target in TARGET_COLUMNS
    }
    for entry in best.values():
        assert "params" in entry
        for key in ["val_r2", "val_mse", "val_rmse", "val_mae"]:
            assert key in entry["metrics"]


def test_tune_models_cv_raises_when_pool_smaller_than_n_splits():
    df = _make_fixture(n=3)

    with pytest.raises(ValueError, match="Fold"):
        tune_models_cv(df, n_splits=5)


def test_tune_models_cv_is_deterministic_for_fixed_random_state():
    df = _make_fixture()

    best_1 = tune_models_cv(df, random_state=42)
    best_2 = tune_models_cv(df, random_state=42)

    assert best_1 == best_2


def test_cv_train_errors_are_from_same_fold_model_and_saved(tmp_path, monkeypatch):
    from sklearn.tree import DecisionTreeRegressor
    from module2_prediction import tuning
    from module2_prediction.pipeline import save_selection_report
    from module2_prediction.config import FEATURE_COLUMNS
    # 학습 오차가 0인 트리로 학습, 검증 오차의 분리 집계를 확인함
    monkeypatch.setattr(tuning, 'MODEL_FACTORIES', {'Tree': lambda **kw: DecisionTreeRegressor(random_state=42)})
    monkeypatch.setattr(tuning, 'MODEL_HYPERPARAMETER_GRIDS', {'Tree': [{}]})
    result = tune_models_cv(_make_fixture(), n_splits=3)
    for entry in result.values():
        assert len(entry['fold_metrics']) == 3
        assert all(f['train_rmse'] == pytest.approx(0) for f in entry['fold_metrics'])
        assert entry['metrics']['val_rmse'] > 0
        assert entry['metrics']['rmse_gap'] == pytest.approx(entry['metrics']['val_rmse'])
        assert entry['metrics']['val_rmse_std'] == pytest.approx(
            np.std([f['val_rmse'] for f in entry['fold_metrics']], ddof=1))
    selection = {'targets': {t: {'features': FEATURE_COLUMNS} for t in TARGET_COLUMNS}, 'cv': {'n_splits': 3}}
    selected = {t: ('Tree', t) for t in TARGET_COLUMNS}
    save_selection_report(result, selected, selection, tmp_path)
    assert len(pd.read_csv(tmp_path / 'cv_fold_scores.csv')) == 6
    assert 'train_rmse_std' in pd.read_csv(tmp_path / 'cv_comparison.csv')
