from pathlib import Path

from module2_prediction.config import TARGET_COLUMNS
from module2_prediction.data_loading import load_cleaned_features, split_by_column
from module2_prediction.evaluation import evaluate_final_model
from module2_prediction.tuning import tune_models


def run(cleaned_data_path: str | Path) -> dict:
    """모델 종류·하이퍼파라미터를 Validation으로 선정하고 Test로 1회 평가한다."""
    df = load_cleaned_features(cleaned_data_path)
    train_df, val_df, test_df = split_by_column(df)

    tuned = tune_models(train_df, val_df)

    best_per_target = {
        target: max(
            (key for key in tuned if key[1] == target),
            key=lambda key: tuned[key]["metrics"]["val_r2"],
        )
        for target in TARGET_COLUMNS
    }

    final_results = {}
    for target, (model_name, _) in best_per_target.items():
        params = tuned[(model_name, target)]["params"]
        test_metrics = evaluate_final_model(
            train_df, val_df, test_df,
            model_name=model_name, model_params=params,
            target_columns=[target],
        )[target]

        final_results[target] = {
            "model": model_name,
            "params": params,
            "val_r2": tuned[(model_name, target)]["metrics"]["val_r2"],
            **test_metrics,
        }

    return final_results


if __name__ == "__main__":
    results = run(Path("data/processed/cleaned_features.csv"))
    print("모듈 2 최종 모델 선정 결과:")
    for target, info in results.items():
        print(f"  {target}: {info}")
