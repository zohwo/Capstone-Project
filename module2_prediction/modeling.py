import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from xgboost import XGBRegressor

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS
from module1_preprocessing.transformation import validate_targets, prepare_split, validate_model_features


MODEL_FACTORIES = {
    "RandomForest": lambda **params: RandomForestRegressor(
        random_state=42,
        n_jobs=1,
        **params,
    ),
    "XGBoost": lambda **params: XGBRegressor(
        random_state=42,
        verbosity=0,
        n_jobs=1,
        **params,
    ),
    "LightGBM": lambda **params: LGBMRegressor(
        random_state=42,
        verbose=-1,
        importance_type="gain",
        n_jobs=1,
        **params,
    ),
    # 표준화는 fit_selected_model()이 선택 피처(원료등급 포함 여부)에 맞춰 적용
    "ElasticNet": lambda **params: ElasticNet(random_state=42, **params),
    "SVR": lambda **params: SVR(kernel="rbf", **params),
}


def target_features(feature_columns, target):
    """공통 피처 목록 또는 타깃별 목록에서 해당 타깃의 입력 컬럼을 반환"""
    return feature_columns[target] if isinstance(feature_columns, dict) else feature_columns


def validate_selected_features(frame, columns):
    """선택된 숫자 피처의 유효성과 원료등급 값을 검사"""
    numeric = [column for column in columns if column != "원료등급"]
    validate_model_features(frame, numeric)
    if "원료등급" in columns:
        if "원료등급" not in frame or not frame["원료등급"].isin(["A", "B", "C"]).all():
            raise ValueError("선택된 원료등급은 결측 없는 A/B/C 값이어야 함")


def fit_selected_model(model, train_df, columns, target):
    """선택 피처로 모델을 학습, SVR/ElasticNet 수치형 표준화, 원료등급 선택 시 원-핫 포함

    - 수치형 표준화는 SVR/ElasticNet 전용, 트리 모델은 스케일 불변이라 불필요
    - 원료등급 선택 시 원-핫 분기 분리, 수치형 표준화와 무관하게 더미(0/1) 그대로 유지
      대신 더미 표준화 시 희귀 등급일수록 0/1 간격 확대, SVR 거리, ElasticNet 정규화가
      등급 빈도에 좌우되는 부작용 방지
    """
    validate_selected_features(train_df, columns)
    uses_scaler = isinstance(model, (SVR, ElasticNet))
    uses_grade = "원료등급" in columns
    if uses_scaler or uses_grade:
        numeric_columns = [col for col in columns if col != "원료등급"]
        transformers = [
            ("numeric", StandardScaler() if uses_scaler else "passthrough", numeric_columns),
        ]
        if uses_grade:
            transformers.append(
                ("grade", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["원료등급"])
            )
        model = Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])
    model.fit(train_df[columns], train_df[target])
    return model


def _evaluate_prepared(model, train_df, val_df, feature_columns, target):
    """전처리된 입력과 타깃을 검증하고 모델 학습 후 학습 RMSE와 검증 지표를 반환"""
    validate_targets(train_df, [target])
    validate_targets(val_df, [target])
    columns = target_features(feature_columns, target)
    validate_selected_features(val_df, columns)
    model = fit_selected_model(model, train_df, columns, target)
    preds = model.predict(val_df[columns])
    mse = mean_squared_error(val_df[target], preds)
    train_preds = model.predict(train_df[columns])
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(train_df[target], train_preds))),
        "val_r2": float(r2_score(val_df[target], preds, force_finite=False)),
        "val_mse": float(mse),
        "val_rmse": float(np.sqrt(mse)),
        "val_mae": float(mean_absolute_error(val_df[target], preds)),
    }


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_params: dict | None = None,
    feature_columns: list[str] | dict[str, list[str]] = FEATURE_COLUMNS,
    target_columns: list[str] = TARGET_COLUMNS,
) -> dict:
    """Train 기준으로 학습, 검증 데이터를 전처리하고 모델별 성능을 비교

    지정한 모델별 파라미터 또는 기본값으로 학습하고 모델, 타깃별
    train_rmse, val_r2, val_mse, val_rmse, val_mae를 반환
    """

    model_params = model_params or {}

    if len(val_df) < 2:
        raise ValueError("Validation 샘플이 2개 미만이면 R²를 계산할 수 없습니다.")

    train_prepared = prepare_split(train_df, train_df)
    val_prepared = prepare_split(train_df, val_df)

    results = {}

    for model_name, make_model in MODEL_FACTORIES.items():
        params = model_params.get(model_name, {})

        for target in target_columns:
            model = make_model(**params)

            results[(model_name, target)] = _evaluate_prepared(
                model=model,
                train_df=train_prepared,
                val_df=val_prepared,
                feature_columns=feature_columns,
                target=target,
            )

    return results
