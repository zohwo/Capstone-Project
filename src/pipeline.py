from pathlib import Path

import pandas as pd

from src.preprocessing import load_raw, clean_schema
from src.imputation import flag_outliers, impute_missing
from src.features import engineer_features
from src.validation import check_ranges


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 로드해 정제·대치·특성공학·검증까지 수행하고 CSV로 저장한다."""
    df = load_raw(input_path)
    df = clean_schema(df)
    # is_duplicate_batch로 플래그된 배치ID 중복 쌍은 날짜·등급·회수율까지 완전히 동일한
    # 복사본으로 확인됨 (분석 완료) — 그대로 두면 교차검증 시 같은 데이터 포인트가
    # train/test 양쪽에 걸쳐 들어가 성능 지표가 부풀려질 수 있어 첫 번째 occurrence만 남긴다.
    # 남은 행의 is_duplicate_batch=True는 "중복이 있었고 하나는 제거됐다"는 감사 기록으로 유지된다.
    df = df.drop_duplicates(subset="배치ID", keep="first")
    df = flag_outliers(df)
    df = impute_missing(df)
    df = engineer_features(df)

    violations = check_ranges(df)
    if violations:
        print(f"경고: 정상범위 위반 발견 - {violations}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    run(
        Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"),
        Path("data/processed/cleaned_features.csv"),
    )
