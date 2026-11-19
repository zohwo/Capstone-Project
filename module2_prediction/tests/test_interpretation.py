import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from module2_prediction.interpretation import compute_shap_values, get_feature_importance


def _make_fixture(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
    })
    y = X["feature_a"] * 2 + rng.normal(0, 0.1, n)
    return X, y


def test_get_feature_importance_returns_sorted_series():
    X, y = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(X, y)

    importance = get_feature_importance(model, feature_columns=["feature_a", "feature_b"])

    assert importance.iloc[0] >= importance.iloc[1]
    assert importance["feature_a"] > importance["feature_b"]


def test_get_feature_importance_normalizes_to_sum_to_one_regardless_of_model():
    X, y = _make_fixture()
    rf_model = RandomForestRegressor(random_state=42).fit(X, y)
    lgbm_model = LGBMRegressor(
        random_state=42, verbose=-1, importance_type="gain", min_child_samples=1,
    ).fit(X, y)

    rf_importance = get_feature_importance(rf_model, feature_columns=["feature_a", "feature_b"])
    lgbm_importance = get_feature_importance(lgbm_model, feature_columns=["feature_a", "feature_b"])

    assert rf_importance.sum() == pytest.approx(1.0)
    assert lgbm_importance.sum() == pytest.approx(1.0)


def test_get_feature_importance_returns_zero_vector_when_total_importance_is_zero():
    class ZeroImportanceModel:
        feature_importances_ = np.array([0.0, 0.0])

    importance = get_feature_importance(ZeroImportanceModel(), feature_columns=["feature_a", "feature_b"])

    assert not importance.isna().any()
    assert (importance == 0).all()


def test_get_feature_importance_raises_for_unsupported_model_without_data():
    class DummyModel:
        pass

    with pytest.raises(ValueError):
        get_feature_importance(DummyModel(), feature_columns=["feature_a"])


def test_get_feature_importance_falls_back_to_permutation_importance_for_pipeline():
    X, y = _make_fixture()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(random_state=42)),
    ]).fit(X, y)

    importance = get_feature_importance(
        model, feature_columns=["feature_a", "feature_b"], X=X, y=y,
    )

    assert importance.sum() == pytest.approx(1.0)
    assert importance["feature_a"] > importance["feature_b"]


def test_compute_shap_values_returns_array_matching_input_rows():
    X, y = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(X, y)

    shap_values = compute_shap_values(model, X)

    assert shap_values.shape[0] == len(X)


def test_compute_shap_values_falls_back_to_generic_explainer_for_pipeline():
    X, y = _make_fixture()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf")),
    ]).fit(X, y)

    shap_values = compute_shap_values(model, X)

    assert shap_values.shape[0] == len(X)


def test_shap_fixed_background_is_additive_and_independent_of_explanation_rows():
    from sklearn.linear_model import LinearRegression
    X, y = _make_fixture()
    model = LinearRegression().fit(X, y)
    background, heldout = X.iloc[:20], X.iloc[20:]
    values = compute_shap_values(model, heldout, background=background)
    np.testing.assert_allclose(values.sum(axis=1),
                               model.predict(heldout) - model.predict(background).mean(), atol=1e-8)
    single = compute_shap_values(model, heldout.iloc[:1], background=background)
    np.testing.assert_allclose(single[0], values[0], atol=1e-8)
    with pytest.raises(ValueError, match='컬럼'):
        compute_shap_values(model, heldout, background=background.iloc[:, ::-1])
