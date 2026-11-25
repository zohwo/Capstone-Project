"""개발 데이터의 층화 CV를 수행하고 평균 검증 RMSE로 모델별 최적 파라미터 선택"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from module1_preprocessing.transformation import validate_targets, prepare_split
from module2_prediction.config import FEATURE_COLUMNS, MODEL_HYPERPARAMETER_GRIDS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES, _evaluate_prepared


def _validate_development_data(df, target_columns):
    """split 컬럼이 있으면 개발 데이터 여부를 검사하고 타깃 값을 검증"""
    if "split" in df and not df["split"].isin(["train", "validation"]).all():
        raise ValueError("모델 선택에는 개발 데이터만 허용됩니다. Test를 제외하세요.")
    validate_targets(df, target_columns)


def _search(prepared_folds, feature_columns, target_columns):
    """동일한 fold에서 후보별 성능을 집계하고 모델, 타깃별 최소 RMSE 후보 반환"""
    best = {}
    for model_name, grid in MODEL_HYPERPARAMETER_GRIDS.items():
        for params in grid:
            for target in target_columns:
                scores = [
                    _evaluate_prepared(MODEL_FACTORIES[model_name](**params), train, val,
                                       feature_columns, target)
                    for train, val in prepared_folds
                ]
                if not all(np.isfinite(m["val_rmse"]) for m in scores):
                    continue
                mse = float(np.mean([m["val_mse"] for m in scores]))
                metrics = {
                    "val_r2": float(np.mean([m["val_r2"] for m in scores])),
                    "val_mse": mse,
                    "val_rmse": float(np.mean([m["val_rmse"] for m in scores])),
                    "val_rmse_std": float(np.std([m["val_rmse"] for m in scores], ddof=1)) if len(scores) > 1 else 0.0,
                    "val_mae_std": float(np.std([m["val_mae"] for m in scores], ddof=1)) if len(scores) > 1 else 0.0,
                    "val_mae": float(np.mean([m["val_mae"] for m in scores])),
                }
                train_scores = [m["train_rmse"] for m in scores]
                metrics["train_rmse"] = float(np.mean(train_scores))
                metrics["train_rmse_std"] = float(np.std(train_scores, ddof=1)) if len(scores) > 1 else 0.0
                metrics["rmse_gap"] = metrics["val_rmse"] - metrics["train_rmse"]
                key = (model_name, target)
                if key not in best or metrics["val_rmse"] < best[key]["metrics"]["val_rmse"]:
                    best[key] = {"params": params, "metrics": metrics,
                                 "fold_metrics": [dict(fold=i + 1, **m) for i, m in enumerate(scores)]}
    missing = [(name, target) for name in MODEL_HYPERPARAMETER_GRIDS
               for target in target_columns if (name, target) not in best]
    if missing:
        raise ValueError(f"유효한 val_rmse를 얻은 후보가 없는 모델, 타깃 조합: {missing}")
    return best


def tune_models_cv(pool_df, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS,
                   n_splits=5, stratify_column="원료등급", random_state=42):
    """개발 데이터를 층화 분할하고 각 fold의 학습 데이터 기준으로 전처리하여 후보 탐색"""
    if n_splits < 2 or len(pool_df) < n_splits:
        raise ValueError(f"{n_splits}-Fold CV를 수행할 표본이 부족합니다.")
    _validate_development_data(pool_df, target_columns)
    if pool_df[stratify_column].isna().any() or pool_df[stratify_column].value_counts().min() < n_splits:
        raise ValueError("층화 CV에는 등급별로 n_splits개 이상의 표본이 필요합니다.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_pos, val_pos in cv.split(pool_df, pool_df[stratify_column]):
        train, val = pool_df.iloc[train_pos], pool_df.iloc[val_pos]
        folds.append((prepare_split(train, train), prepare_split(train, val)))
    return _search(folds, feature_columns, target_columns)
