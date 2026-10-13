from pathlib import Path

import pandas as pd

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import flag_outliers, impute_missing_from_reference, interpolate_analysis_date
from src.preprocessing import clean_schema, load_raw
from src.splitting import temporal_train_val_test_split
from src.validation import check_ranges, run_ablation_test


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 로드해 정제·분할·대치·특성공학·검증까지 수행하고 CSV로 저장한다.

    Train/Validation/Test로 시간순 분할한 뒤, 결측치 대치 통계(중앙값·회귀계수)는
    Train 구간으로만 계산해 Validation/Test에 적용한다 — data leakage 없이 미래
    배치를 평가하는 `src/validation.py`의 `run_ablation_test`와 동일한 원칙이다.
    """
    df = load_raw(input_path)
    df = clean_schema(df)
    # is_duplicate_batch로 플래그된 배치ID 중복 쌍은 날짜·등급·회수율까지 완전히 동일한
    # 복사본으로 확인됨 (분석 완료) — 그대로 두면 교차검증 시 같은 데이터 포인트가
    # train/test 양쪽에 걸쳐 들어가 성능 지표가 부풀려질 수 있어 첫 번째 occurrence만 남긴다.
    # 남은 행의 is_duplicate_batch=True는 "중복이 있었고 하나는 제거됐다"는 감사 기록으로 유지된다.
    df = df.drop_duplicates(subset="배치ID", keep="first")
    df = flag_outliers(df)

    pre_impute_df = df.copy()

    train_i, val_i, test_i = temporal_train_val_test_split(df)
    df["split"] = "train"
    df.loc[val_i, "split"] = "validation"
    df.loc[test_i, "split"] = "test"

    train_df = df.loc[train_i]
    df = impute_missing_from_reference(train_df, df)
    # 분석일자는 모델 피처로 쓰이지 않는 위치 기반 보간이라 train/val/test 구분 없이
    # 배치ID 순서 전체로 처리한다 (impute_missing_from_reference와 별개).
    df = interpolate_analysis_date(df)
    df = engineer_features(df)

    violations = check_ranges(df)
    if violations:
        print(f"경고: 정상범위 위반 발견 - {violations}")

    base_features = COMPOSITION_COLUMNS + PROCESS_COLUMNS
    ablation_results = run_ablation_test(
        pre_impute_df,
        {"base": base_features, "base_plus_derived": base_features + DERIVED_FEATURES},
        TARGET_COLUMNS,
    )
    print("파생변수 전/후 성능 비교 (Validation/Test 기준 R²·RMSE·MAE):")
    for (case_name, target), metrics in ablation_results.items():
        print(f"  {case_name} / {target}: {metrics}")

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
