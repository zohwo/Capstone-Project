import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES, target_features, fit_selected_model, validate_selected_features
from module1_preprocessing.transformation import validate_targets, prepare_split


def evaluate_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict,
    feature_columns: list = FEATURE_COLUMNS,
    target_columns: list = TARGET_COLUMNS,
    output_dir=None,
) -> dict:
    """선택된 모델(model_name, model_params)로 Train+Validation 합쳐 재학습, Test에 1회 적용

    - 대치 기준은 Train+Validation 원본에서 재계산해 Test에 동일 적용
    - test_r2는 force_finite=False — 상수 타깃이면 sklearn 기본값(0.0 clip) 대신 NaN/-inf 그대로 반환
    - output_dir 지정 시: 모델 저장 후 재로드해 예측 일치 검증(불일치 시 ValueError), 예측값 CSV 저장
    """
    if len(test_df) < 2:
        raise ValueError("Test 샘플이 2개 미만이면 R²를 계산할 수 없습니다.")

    for frame in (train_df, val_df, test_df):
        validate_targets(frame, target_columns)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_prepared = prepare_split(train_val_df, train_val_df)
    test_prepared = prepare_split(train_val_df, test_df)
    make_model = MODEL_FACTORIES[model_name]

    results = {}
    for target in target_columns:
        columns = target_features(feature_columns, target)
        validate_selected_features(test_prepared, columns)
        model = make_model(**model_params)
        model = fit_selected_model(model, train_prepared, columns, target)
        preds = model.predict(test_prepared[columns])

        if output_dir is not None:
            from pathlib import Path
            from module2_prediction.artifacts import save_bundle, load_bundle, predict_bundle
            path = Path(output_dir) / f"{target}.joblib"
            save_bundle(path, model, train_val_df, columns, target)
            reloaded = predict_bundle(load_bundle(path), test_df)
            if not np.allclose(preds, reloaded, rtol=1e-12, atol=1e-12):
                raise ValueError("저장 전후 모델 예측이 일치하지 않음")
            pd.DataFrame({"batch_id": test_df.get("배치ID", test_df.index),
                          "actual": test_df[target], "prediction": preds}).to_csv(
                Path(output_dir) / f"{target}_predictions.csv", index=False)

        test_r2 = float(r2_score(test_df[target], preds, force_finite=False))
        mse = mean_squared_error(test_df[target], preds)
        results[target] = {
            "test_r2": test_r2,
            "test_mse": float(mse),
            "test_rmse": float(np.sqrt(mse)),
            "test_mae": float(mean_absolute_error(test_df[target], preds)),
        }

    return results
