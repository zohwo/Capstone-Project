import hashlib
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from module1_preprocessing.config import (
    COMPOSITION_COLUMNS,
    BASE_FEATURES,
    DERIVED_FEATURES,
    REFERENCE_EXCEPTION_RANGES,
    PROCESS_COLUMNS,
    TARGET_COLUMNS,
)
from module1_preprocessing.features import engineer_features
from module1_preprocessing.transformation import validate_targets, validate_model_features
from module1_preprocessing.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date
from module1_preprocessing.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from module1_preprocessing.splitting import stratified_train_val_test_split
from module1_preprocessing.validation import check_ranges, format_range_violations, check_missing, run_cv_ablation_test, check_composition_sum


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 정제·분할·대치·특성공학·검증한 뒤 CSV로 저장

    원료등급 구성비를 보존하는 stratified 분할로 Train/Validation/Test를 나누고, 
    Train 기준 대치 결과는 품질 확인용으로 저장 모델링용 대치 전 데이터와 개발 데이터 비교 후 승인된 공통 피처 구성을 별도 저장하며
    Test는 평가하지 않음
    """

    df = load_raw(input_path)
    df = clean_schema(df)

    raw_violations = check_ranges(df)
    print("데이터 정상범위 이탈 현황:")
    print(format_range_violations(df, raw_violations))

    validate_targets(df, TARGET_COLUMNS)
    df = resolve_duplicate_batches(df)

    conflicting = df.loc[df["is_conflicting_duplicate"], "배치ID"].unique().tolist()
    if conflicting:
        raise ValueError(
            f"충돌 중복 배치ID가 있습니다 - 같은 배치ID인데 값이 다른 행이 존재합니다 "
            f"(사람 확인 필요, 해결 전까지 분할하면 같은 배치가 Train/Test에 걸쳐 들어가 "
            f"데이터 누수가 생길 수 있음): {conflicting}"
        )

    df = replace_outliers_with_nan(df)

    # 동일 시료·동일 질량 기준 wt%라는 전제. 대치 전 판정도 보존함
    composition_before = check_composition_sum(df).add_prefix("pre_impute_")
    df = df.join(composition_before)

    missing = check_missing(df, columns=["분석일자"] + BASE_FEATURES + TARGET_COLUMNS)
    print(f"결측치 확인: {missing if missing else '없음'}")

    pre_impute_df = df.copy()
    df = interpolate_analysis_date(df)

    train_i, val_i, test_i = stratified_train_val_test_split(df)

    df["split"] = "train"
    df.loc[val_i, "split"] = "validation"
    df.loc[test_i, "split"] = "test"

    pre_impute_df["split"] = df["split"]
    train_df = df.loc[train_i].copy()

    df = impute_missing_from_reference(train_df, df)

    for col in ["원료투입량_kg", "황산투입량_L"]:
        bad_batches = df.loc[df[col] <= 0, "배치ID"].tolist()
        if bad_batches:
            raise ValueError(
                f"{col}이(가) 0 이하인 배치가 있어 acid_to_ore_ratio/"
                f"feed_impurity_g_per_l을 계산할 수 없습니다: {bad_batches}"
            )

    df = engineer_features(df)
    df = df.join(check_composition_sum(df))
    for stage, column in (("대치 전", "pre_impute_composition_sum_review_flag"),
                          ("대치 후", "composition_sum_review_flag")):
        batches = df.loc[df[column].fillna(False), "배치ID"].tolist()
        print(f"{stage} 성분 합계 100% 초과 검토 대상: {batches or '없음'} (자동 보정 없음)")

    base_features = BASE_FEATURES

    feature_columns = base_features + DERIVED_FEATURES
    remaining_missing = check_missing(df, columns=feature_columns + TARGET_COLUMNS)

    if remaining_missing:
        raise ValueError(f"정제 후 결측치 발견 - {remaining_missing}")

    validate_model_features(df, feature_columns)
    print("정제 후 남은 결측치: 없음")

    violations = check_ranges(df, reference_exception_ranges=REFERENCE_EXCEPTION_RANGES)
    if violations:
        raise ValueError(f"정상범위와 문헌 기반 보존 예외를 벗어난 값이 남아 있습니다 - {format_range_violations(df, violations)}")
    print("정제 후 보존 기준 위반: 없음")

    select_and_save_features(pre_impute_df, output_path)
    df.to_csv(output_path, index=False)

    return df


def evaluate_feature_selection(pre_impute_df: pd.DataFrame) -> tuple[dict, dict]:
    """고정 split에서 네 조합을 한 번 비교하고 선택 정보·평가 결과 반환"""
    validate_targets(pre_impute_df, TARGET_COLUMNS)
    if set(pre_impute_df["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("고정 train/validation/test split이 필요합니다.")
    if pre_impute_df["배치ID"].isna().any() or pre_impute_df["배치ID"].duplicated().any():
        raise ValueError("배치ID는 결측 없이 유일해야 합니다.")
    base_features = BASE_FEATURES
    feature_sets = {
        "base": base_features,
        "base_plus_grade": base_features + ["원료등급"],
        "base_plus_derived": base_features + DERIVED_FEATURES,
        "all_features": base_features + ["원료등급"] + DERIVED_FEATURES,
    }

    cv_pool = pd.concat([pre_impute_df.loc[pre_impute_df["split"] == name]
                         for name in ("train", "validation")], ignore_index=True)

    model_factory = lambda random_state: LGBMRegressor(random_state=random_state, verbose=-1, n_jobs=1)

    ablation_results = run_cv_ablation_test(
        cv_pool,
        feature_sets,
        TARGET_COLUMNS,
        model_factory=model_factory,
    )

    print("파생변수 전/후 성능 비교 (5-Fold CV 평균 R²·MSE·RMSE·MAE):")
    for (case_name, target), metrics in ablation_results.items():
        print(f"  {case_name} / {target}: {metrics}")

    # 최소 MAE 조합은 참고용으로 기록하며 승인된 공통 구성을 유지함
    lowest_mae_cases = {
        target: min(feature_sets, key=lambda case: ablation_results[(case, target)]["cv_mae"])
        for target in TARGET_COLUMNS
    }

    best_cases = {target: "base_plus_derived" for target in TARGET_COLUMNS}

    for target, best_case in best_cases.items():
        print(f"{target} 최종 피처 조합: {best_case}")

    selection = {
        "schema_version": 2,
        "preprocessing_stage": "before_imputation",
        "primary_metric": "mae",
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42,
               "stratify_column": "원료등급"},
        "feature_selection_model": "LightGBM",
        "selection_policy": "approved_shared_features",
        "selection_reason": "반복 CV 비교와 입력 구성의 단순성을 고려하여 공통 4종 채택함",
        "selection_basis": {"seeds": [42, 17, 83], "n_splits": 5,
                            "report": "module1_preprocessing/reports/derived_subset_selection/subset_summary.csv"},
        "lowest_mae_cases": lowest_mae_cases,
        "targets": {
            target: {
                "case": best_cases[target],
                "features": feature_sets[best_cases[target]],
                "cv_metrics": ablation_results[(best_cases[target], target)],
            }
            for target in TARGET_COLUMNS
        },
    }
    selection["data_sha256"] = hashlib.sha256(pre_impute_df.to_csv(index=False).encode("utf-8")).hexdigest()
    return selection, ablation_results


def save_feature_selection(pre_impute_df: pd.DataFrame, selection: dict, output_path: str | Path) -> dict:
    """평가한 데이터와 동일한 입력인지 확인하고 재학습 없이 산출물 저장"""
    csv_text = pre_impute_df.to_csv(index=False)
    digest = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    if selection.get("data_sha256") != digest:
        raise ValueError("특징 평가 이후 입력 데이터가 변경됨 — 재평가 필요함")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.parent / "pre_impute_data.csv"
    selection_path = output_path.parent / "feature_selection.json"
    selection = {**selection, "data_file": raw_path.name}
    raw_path.write_bytes(csv_text.encode("utf-8"))
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"모델링 입력: {raw_path}; 선택 결과: {selection_path} (Test 평가 없음)")
    return selection


def select_and_save_features(pre_impute_df: pd.DataFrame, output_path: str | Path) -> dict:
    """고정 split에서 네 조합을 비교하고 승인된 공통 4종 저장"""
    selection, _ = evaluate_feature_selection(pre_impute_df)
    return save_feature_selection(pre_impute_df, selection, output_path)


if __name__ == "__main__":
    run(
        Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"),
        Path("data/processed/cleaned_features.csv"),
    )
    print("데이터 정제 및 특성공학 완료. 결과를 data/processed/cleaned_features.csv에 저장 완료")
