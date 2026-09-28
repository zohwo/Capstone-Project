from pathlib import Path

import pandas as pd

from src.config import NORMAL_RANGES
from src.pipeline import run

RAW_PATH = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")


def test_run_produces_clean_dataframe_with_no_missing_in_key_columns(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    assert output_path.exists()
    assert result.shape[0] == 424
    for col in list(NORMAL_RANGES.keys()) + ["배치ID", "원료등급"]:
        assert result[col].isna().sum() == 0


def test_run_output_has_no_range_violations(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    for col, (low, high) in NORMAL_RANGES.items():
        assert result[col].between(low, high).all(), f"{col}에 범위 위반 존재"


def test_run_adds_all_engineered_columns(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    for col in [
        "acid_to_ore_ratio",
        "leach_intensity",
        "total_impurity_pct",
        "target_purity_ratio",
        "analysis_month",
        "is_duplicate_batch",
    ]:
        assert col in result.columns
