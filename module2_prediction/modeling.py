import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS

MODEL_FACTORIES = {
    "RandomForest": lambda **params: RandomForestRegressor(random_state=42, **params),
    "XGBoost": lambda **params: XGBRegressor(random_state=42, verbosity=0, **params),
    "LightGBM": lambda **params: LGBMRegressor(random_state=42, verbose=-1, **params),
}


def _fit_and_evaluate(model, train_df, val_df, feature_columns, target) -> dict:
    """model을 train_df로 학습해 val_df로 평가한다 (Validation 지표만)."""
    model.fit(train_df[feature_columns], train_df[target])
    preds = model.predict(val_df[feature_columns])

    return {
        "val_r2": float(r2_score(val_df[target], preds)),
        "val_rmse": float(np.sqrt(mean_squared_error(val_df[target], preds))),
        "val_mae": float(mean_absolute_error(val_df[target], preds)),
    }


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_params: dict | None = None,
    feature_columns: list = FEATURE_COLUMNS,
    target_columns: list = TARGET_COLUMNS,
) -> dict:
    """모델 종류별로 Validation 성능을 비교한다 (feature는 고정, Test는 안 봄).

    model_params: {모델명: 하이퍼파라미터 dict}. 생략하면 각 모델 기본값을 쓴다.
    반환값: {(모델명, target): {val_r2, val_rmse, val_mae}}
    """
    model_params = model_params or {}

    for target in target_columns:
        if train_df[target].isna().any() or val_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    results = {}
    for model_name, make_model in MODEL_FACTORIES.items():
        params = model_params.get(model_name, {})
        for target in target_columns:
            model = make_model(**params)
            results[(model_name, target)] = _fit_and_evaluate(
                model, train_df, val_df, feature_columns, target
            )

    return results
