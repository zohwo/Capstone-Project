from pathlib import Path

import pandas as pd

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import flag_outliers, impute_missing_from_reference, interpolate_analysis_date
from src.preprocessing import clean_schema, load_raw
from src.splitting import temporal_train_val_test_split
from src.validation import check_ranges, check_missing, run_ablation_test


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 로드해 정제·분할·대치·특성공학·검증까지 수행하고 CSV로 저장한다.

    Train/Validation/Test로 시간순 분할한 뒤, 결측치 대치 통계(중앙값·회귀계수)는
    Train 구간으로만 계산해 Validation/Test에 적용한다 — data leakage 없이 미래
    배치를 평가하는 `src/validation.py`의 `run_ablation_test`와 동일한 원칙이다.
    """
    df = load_raw(input_path)

    raw_violations = check_ranges(df)
    print(f"raw 데이터 정상범위 위반: {raw_violations if raw_violations else '없음'}")

    df = clean_schema(df)
    df = df.drop_duplicates(subset="배치ID", keep="first")
    df = flag_outliers(df)

    missing = check_missing(df)
    print(f"결측치 확인: {missing if missing else '없음'}")

    pre_impute_df = df.copy()

    df = interpolate_analysis_date(df)

    train_i, val_i, test_i = temporal_train_val_test_split(df)
    df["split"] = "train"
    df.loc[val_i, "split"] = "validation"
    df.loc[test_i, "split"] = "test"

    train_df = df.loc[train_i]
    df = impute_missing_from_reference(train_df, df)
    df = engineer_features(df)

    # flag_outliers가 범위 밖 값을 전부 NaN으로 치환하고 대치까지 끝난 상태이므로,
    # 여기서 위반이 발견되면 대치/파생변수 로직에 버그가 있다는 뜻이다 — 잘못된
    # 데이터를 CSV로 내보내지 않도록 경고만 찍지 않고 즉시 중단한다.
    violations = check_ranges(df)
    assert not violations, f"정상범위 위반 발견 - {violations}"
    print("정제 후 정상범위 위반: 없음")

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
