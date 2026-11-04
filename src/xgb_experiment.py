"""
실행: python -m src.xgb_experiment
- 팀 파이프라인과 같은 정제·시간순 분할·Train 기준 대치를 그대로 사용한다.
- RandomForest와 XGBoost를 같은 조건에서 비교하고, 선택 기준은 Validation RMSE(최소)이다.
- 결과는 data/processed/model_comparison.csv 에 저장한다 (data/ 는 git에 포함되지 않음).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.imputation import interpolate_analysis_date, replace_outliers_with_nan
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split
from src.validation import _prepare_split

DEFAULT_INPUT = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")
DEFAULT_OUTPUT = Path("data/processed/model_comparison.csv")


def make_model(model_name: str, random_state: int = 42):
    """'rf' 또는 'xgb' 회귀 모델을 만든다. XGBoost 값은 튜닝 전 초기값이다."""

    if model_name == "rf":
        return RandomForestRegressor(random_state=random_state)

    if model_name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _features(prepared: pd.DataFrame, columns: list, ref_columns=None) -> pd.DataFrame:
    x = pd.get_dummies(prepared[columns])
    return x if ref_columns is None else x.reindex(columns=ref_columns, fill_value=0)


def run_experiment(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, model_names=("rf", "xgb")) -> pd.DataFrame:
    """모델별 Validation 비교 → Validation RMSE로 피처 조합 선택 → Test 1회 평가."""

    # 팀 pipeline.run 과 동일한 정제·분할 (대치는 분할 후 Train 통계)
    df = load_raw(input_path)
    df = clean_schema(df)
    df = resolve_duplicate_batches(df)
    df = replace_outliers_with_nan(df)
    df = interpolate_analysis_date(df)

    train_i, val_i, test_i = temporal_train_val_test_split(df)
    train_df, val_df, test_df = df.loc[train_i], df.loc[val_i], df.loc[test_i]
    print(f"분할: train {len(train_df)} / validation {len(val_df)} / test {len(test_df)}")

    base = COMPOSITION_COLUMNS + PROCESS_COLUMNS
    feature_sets = {"base": base, "base_plus_derived": base + DERIVED_FEATURES}

    train_prep = _prepare_split(train_df, train_df)
    val_prep = _prepare_split(train_df, val_df)

    train_val_df = pd.concat([train_df, val_df])
    tv_prep = _prepare_split(train_val_df, train_val_df)
    test_prep = _prepare_split(train_val_df, test_df)

    rows = []
    for model_name in model_names:
        print(f"\n===== 모델: {model_name} =====")
        for target in TARGET_COLUMNS:
            val_scores = {}
            for case, cols in feature_sets.items():
                x_tr = _features(train_prep, cols)
                x_val = _features(val_prep, cols, x_tr.columns)
                model = make_model(model_name).fit(x_tr, train_df[target])
                val_scores[case] = _metrics(val_df[target], model.predict(x_val))
                print(f"  [Validation] {target} / {case}: {val_scores[case]}")

            best = min(val_scores, key=lambda c: val_scores[c]["rmse"])
            cols = feature_sets[best]
            x_tv = _features(tv_prep, cols)
            x_te = _features(test_prep, cols, x_tv.columns)
            model = make_model(model_name).fit(x_tv, train_val_df[target])
            test_scores = _metrics(test_df[target], model.predict(x_te))
            print(f"  [Test] {target} / 선택 피처 {best}: {test_scores}")

            for case, m in val_scores.items():
                rows.append({"model": model_name, "target": target, "feature_set": case,
                             "stage": "validation", "selected": case == best, **m})
            rows.append({"model": model_name, "target": target, "feature_set": best,
                         "stage": "test", "selected": True, **test_scores})

    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_path}")
    return result


if __name__ == "__main__":
    run_experiment()
