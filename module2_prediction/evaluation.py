import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES


def evaluate_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict,
    feature_columns: list = FEATURE_COLUMNS,
    target_columns: list = TARGET_COLUMNS,
) -> dict:
    """선택된 모델(model_name, model_params)로 Train+Validation을 합쳐 재학습한 뒤 Test에 딱 한 번 적용한다."""
    for target in target_columns:
        if (
            train_df[target].isna().any()
            or val_df[target].isna().any()
            or test_df[target].isna().any()
        ):
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    make_model = MODEL_FACTORIES[model_name]

    results = {}
    for target in target_columns:
        model = make_model(**model_params)
        model.fit(train_val_df[feature_columns], train_val_df[target])
        preds = model.predict(test_df[feature_columns])

        results[target] = {
            "test_r2": float(r2_score(test_df[target], preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(test_df[target], preds))),
            "test_mae": float(mean_absolute_error(test_df[target], preds)),
        }

    return results
