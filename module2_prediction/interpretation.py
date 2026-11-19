import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from module2_prediction.config import FEATURE_COLUMNS


def get_feature_importance(
    model,
    feature_columns: list = FEATURE_COLUMNS,
    X: pd.DataFrame = None,
    y: pd.Series = None,
) -> pd.Series:
    """feature명과 묶은 중요도를 내림차순 반환 (합 1로 정규화)

    - model에 feature_importances_ 속성 있으면 그대로 사용
      - 합이 1 아니면 정규화, 전부 0이면 나누지 않고 그대로 반환(0으로 나누면 전부 NaN)
    - 속성 없고 X, y 주어지면 permutation importance로 대체, 음수는 0으로 절단 후 정규화
    - 둘 다 아니면 ValueError
    """
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        total = raw.sum()
        normalized = raw / total if total > 0 else raw
        return pd.Series(normalized, index=feature_columns).sort_values(ascending=False)

    if X is not None and y is not None:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        raw = result.importances_mean.clip(min=0)
        total = raw.sum()
        normalized = raw / total if total > 0 else raw
        return pd.Series(normalized, index=feature_columns).sort_values(ascending=False)

    raise ValueError(f"{type(model).__name__}은 feature_importances_를 지원하지 않습니다.")


def compute_shap_values(model, X: pd.DataFrame, background: pd.DataFrame = None):
    """model.predict 기반 예측 기여(SHAP) 계산

    - model에 feature_importances_ 속성 있으면 TreeExplainer 사용(background 주어지면 interventional perturbation)
    - 속성 없으면 model.predict를 감싼 범용 Explainer 사용
    - background 생략 시 X를 배경으로도 사용
    - background 주어지면 컬럼, 순서가 X와 같아야 함(다르면 ValueError)
    """
    if background is not None and list(background.columns) != list(X.columns):
        raise ValueError("SHAP 배경과 설명 대상의 컬럼, 순서가 같아야 합니다.")
    if hasattr(model, "feature_importances_"):
        kwargs = {} if background is None else dict(
            data=background, feature_perturbation="interventional")
        explainer = shap.TreeExplainer(model, **kwargs)
        return explainer.shap_values(X)

    reference = X if background is None else background
    # sklearn 이름 있는 입력 순서 보존, ndarray 기반 SHAP 호출도 지원
    def predict(values):
        return model.predict(pd.DataFrame(values, columns=X.columns))

    explainer = shap.Explainer(predict, reference, seed=42)
    return explainer(X).values
