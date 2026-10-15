import pandas as pd
import shap

from module2_prediction.config import FEATURE_COLUMNS


def get_feature_importance(model, feature_columns: list = FEATURE_COLUMNS) -> pd.Series:
    """학습된 모델의 feature_importances_를 feature명과 묶어 내림차순으로 반환한다."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__}은 feature_importances_를 지원하지 않습니다.")

    return pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)


def compute_shap_values(model, X: pd.DataFrame):
    """TreeExplainer로 SHAP value를 계산한다 (RandomForest/XGBoost/LightGBM 전용)."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
