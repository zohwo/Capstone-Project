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
