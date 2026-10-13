from pathlib import Path

import pandas as pd

from src.config import NORMAL_RANGES
from src.pipeline import run

RAW_PATH = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")


def test_run_produces_clean_dataframe_with_no_missing_in_key_columns(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    assert output_path.exists()
    assert result.shape[0] == 420
    assert result["배치ID"].is_unique
    # 중복 쌍 4건은 첫 occurrence만 남기고 제거되지만, 남은 행의 is_duplicate_batch는
    # "이 배치는 원래 중복이 있었고 하나는 제거됐다"는 감사 기록으로 True가 유지된다.
    assert result["is_duplicate_batch"].sum() == 4
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
        "is_conflicting_duplicate",
    ]:
        assert col in result.columns


def test_run_adds_temporal_split_column(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    assert "split" in result.columns
    assert set(result["split"].unique()) == {"train", "validation", "test"}
    assert result["split"].value_counts().to_dict() == {
        "train": 294,
        "validation": 63,
        "test": 63,
    }

    # split은 배치ID 순서를 따라야 한다 (train이 가장 이르고 test가 가장 늦음)
    batch_seq = result["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    train_max = batch_seq[result["split"] == "train"].max()
    val_min = batch_seq[result["split"] == "validation"].min()
    val_max = batch_seq[result["split"] == "validation"].max()
    test_min = batch_seq[result["split"] == "test"].min()
    assert train_max < val_min
    assert val_max < test_min
