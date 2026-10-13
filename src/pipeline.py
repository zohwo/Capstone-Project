from pathlib import Path

import pandas as pd

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split
from src.validation import check_ranges, check_missing, evaluate_final_model, run_ablation_test


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 정제·분할·대치·특성공학·검증한 뒤 CSV로 저장한다.

    시간순으로 Train/Validation/Test를 분할하고,
    Train 통계만 사용해 전체 구간의 결측치를 대치하여 데이터 누수를 방지한다.
    """

    df = load_raw(input_path)

    raw_violations = check_ranges(df)
    print(f"raw 데이터 정상범위 위반: {raw_violations if raw_violations else '없음'}")

    df = clean_schema(df)
    df = resolve_duplicate_batches(df)
    df = replace_outliers_with_nan(df)

    missing = check_missing(df)
    print(f"결측치 확인: {missing if missing else '없음'}")

    df = interpolate_analysis_date(df)
    pre_impute_df = df.copy()

    train_i, val_i, test_i = temporal_train_val_test_split(df)

    df["split"] = "train"
    df.loc[val_i, "split"] = "validation"
    df.loc[test_i, "split"] = "test"

    train_df = df.loc[train_i].copy()

    df = impute_missing_from_reference(train_df, df)
    df = engineer_features(df)

    remaining_missing = check_missing(df)

    if remaining_missing:
        raise ValueError(f"정제 후 결측치 발견 - {remaining_missing}")

    print("정제 후 남은 결측치: 없음")

    violations = check_ranges(df)

    if violations:
        raise ValueError(f"정상범위 위반 발견 - {violations}")

    print("정제 후 정상범위 위반: 없음")

    base_features = COMPOSITION_COLUMNS + PROCESS_COLUMNS

    feature_sets = {
        "base": base_features,
        "base_plus_derived": base_features + DERIVED_FEATURES,
    }

    ablation_results = run_ablation_test(
        pre_impute_df.loc[train_i],
        pre_impute_df.loc[val_i],
        feature_sets,
        TARGET_COLUMNS,
    )

    print("파생변수 전/후 성능 비교 (Validation 기준 R²·RMSE·MAE):")
    for (case_name, target), metrics in ablation_results.items():
        print(f"  {case_name} / {target}: {metrics}")

    # 목표변수별로 Validation R²가 가장 높은 피처 조합 선택
    best_cases = {
        target: max(
            feature_sets,
            key=lambda case_name: ablation_results[(case_name, target)]["val_r2"]
        )
        for target in TARGET_COLUMNS
    }

    for target, best_case in best_cases.items():
        print(f"{target} 최종 피처 조합: {best_case}")

    # 각 목표변수별 최종 피처 조합으로 Test 1회 평가
    final_test_results = {}

    for target in TARGET_COLUMNS:
        final_features = feature_sets[best_cases[target]]

        target_result = evaluate_final_model(
            pre_impute_df.loc[train_i],
            pre_impute_df.loc[val_i],
            pre_impute_df.loc[test_i],
            final_features,
            [target],
        )

        final_test_results[target] = target_result[target]

    print("최종 Test 성능:")
    for target, metrics in final_test_results.items():
        print(f"  {target}: {metrics}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    run(
        Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"),
        Path("data/processed/cleaned_features.csv"),
    )
    print("데이터 정제 및 특성공학 완료. 결과를 data/processed/cleaned_features.csv에 저장 완료")
