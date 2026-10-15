import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

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


def test_get_feature_importance_raises_for_unsupported_model():
    class DummyModel:
        pass

    with pytest.raises(ValueError):
        get_feature_importance(DummyModel(), feature_columns=["feature_a"])


def test_compute_shap_values_returns_array_matching_input_rows():
    X, y = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(X, y)

    shap_values = compute_shap_values(model, X)

    assert shap_values.shape[0] == len(X)
