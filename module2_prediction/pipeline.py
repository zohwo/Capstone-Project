from pathlib import Path

import pandas as pd
import json

from module2_prediction.config import TARGET_COLUMNS, PRE_IMPUTE_DATA_PATH
from module2_prediction.data_loading import load_modeling_inputs, split_by_column
from module2_prediction.evaluation import evaluate_final_model
from module2_prediction.tuning import tune_models_cv


def save_selection_report(tuned, best_per_target, selection, output_dir):
    """모델별 CV 비교표, fold 오차, 선정 근거 저장, 비교표 반환"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [dict(model=name, target=target, params=v["params"], **v["metrics"])
               for (name, target), v in tuned.items()]
    pd.DataFrame(records).to_csv(output_dir / "cv_comparison.csv", index=False)
    folds = [dict(model=name, target=target, **fold)
             for (name, target), v in tuned.items() for fold in v["fold_metrics"]]
    pd.DataFrame(folds).to_csv(output_dir / "cv_fold_scores.csv", index=False)
    features = {t: entry["features"] for t, entry in selection["targets"].items()}
    payload = dict(
        primary_metric="rmse", cv=selection["cv"], selected=best_per_target,
        source_sha256=selection.get("data_sha256"), features=features,
        selection_reason="동일한 CV 조건에서 평균 검증 RMSE가 최소인 모델을 타깃별 선정; MAE와 R²는 보조 지표",
        cv_role="CV RMSE는 모델, 하이퍼파라미터 선택에 사용한 점수이므로 독립 평가 점수로 해석하지 않는다",
        test_role="선정 완료 후 평가; Test 결과를 모델, 피처 재선택에 사용하지 않음",
        test_history_caveat="현재 공식 실행에서 Test를 선택에 쓰는 경로는 발견되지 않았고, 과거 피처 선택에 현재 Test 배치가 쓰였는지는 미확인",
        uncertainty="CV 표준편차는 분할별 성능 변동이며 개별 배치 예측구간이 아님",
    )
    (output_dir / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


def run(data_path: str | Path = PRE_IMPUTE_DATA_PATH, selection_path=None, output_dir=None) -> dict:
    """데이터 로드, CV 튜닝, 타깃별 모델 선정, 최종 평가 수행, 지정 경로에 결과 저장"""
    df, selection = load_modeling_inputs(data_path, selection_path)
    features = {target: entry["features"] for target, entry in selection["targets"].items()}
    train_df, val_df, test_df = split_by_column(df)
    cv_pool = pd.concat([train_df, val_df], ignore_index=True)

    cv = selection["cv"]
    tuned = tune_models_cv(cv_pool, feature_columns=features,
                           n_splits=cv["n_splits"], random_state=cv["random_state"],
                           stratify_column=cv["stratify_column"])

    best_per_target = {
        target: min(
            (key for key in tuned if key[1] == target),
            key=lambda key: tuned[key]["metrics"]["val_rmse"],
        )
        for target in TARGET_COLUMNS
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_selection_report(tuned, best_per_target, selection, output_dir)

    final_results = {}
    for target, (model_name, _) in best_per_target.items():
        params = tuned[(model_name, target)]["params"]
        test_metrics = evaluate_final_model(
            train_df, val_df, test_df,
            model_name=model_name, model_params=params,
            target_columns=[target], feature_columns=features[target],
            output_dir=output_dir,
        )[target]

        final_results[target] = {
            "model": model_name,
            "params": params,
            "primary_metric": "rmse",
            "features": features[target],
            **tuned[(model_name, target)]["metrics"],
            **test_metrics,
        }

    if output_dir is not None:
        (output_dir / "results.json").write_text(
            json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return final_results


if __name__ == "__main__":
    results = run(output_dir="module2_prediction/reports/final")
    print("모듈 2 최종 모델 선정 결과:")
    for target, info in results.items():
        print(f"  {target}: {info}")
