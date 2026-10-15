from module2_prediction.config import FEATURE_COLUMNS, MODEL_HYPERPARAMETER_GRIDS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES, _fit_and_evaluate


def tune_models(
    train_df, val_df, feature_columns: list = FEATURE_COLUMNS, target_columns: list = TARGET_COLUMNS
) -> dict:
    """MODEL_HYPERPARAMETER_GRIDS의 각 조합을 Validation으로 평가해 모델·타겟별 최적 하이퍼파라미터를 고른다.

    Test는 전혀 보지 않는다. 반환값: {(모델명, target): {"params": {...}, "metrics": {val_r2, val_rmse, val_mae}}}
    """
    for target in target_columns:
        if train_df[target].isna().any() or val_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    best = {}
    for model_name, grid in MODEL_HYPERPARAMETER_GRIDS.items():
        make_model = MODEL_FACTORIES[model_name]
        for params in grid:
            for target in target_columns:
                model = make_model(**params)
                metrics = _fit_and_evaluate(model, train_df, val_df, feature_columns, target)

                key = (model_name, target)
                if key not in best or metrics["val_r2"] > best[key]["metrics"]["val_r2"]:
                    best[key] = {"params": params, "metrics": metrics}

    return best
