"""모듈 1·2에서 공유하는 학습 데이터 기준 전처리."""
import numpy as np
import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS
from module1_preprocessing.features import engineer_features
from module1_preprocessing.imputation import impute_missing_from_reference

RAW_INPUT_COLUMNS = COMPOSITION_COLUMNS + PROCESS_COLUMNS + [
    "원료등급", "원료투입량_kg", "황산투입량_L", "분석일자",
]


def prepare_split(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """reference_df만으로 대치 기준을 계산하여 target_df에 적용한 뒤 파생변수 생성

    CV에서는 reference_df가 반드시 해당 fold의 학습 부분이어야 함
    분석일자는 탐색용이며 모델 입력에는 포함하지 않음
    """
    for name, frame in (("reference", reference_df), ("target", target_df)):
        missing = set(RAW_INPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 데이터에 전처리 원본 컬럼이 없습니다: {sorted(missing)}")
    out = impute_missing_from_reference(reference_df, target_df)
    out["분석일자"] = pd.to_datetime(out["분석일자"], errors="coerce")
    return engineer_features(out)


def validate_model_features(frame: pd.DataFrame, columns: list[str]) -> None:
    """전처리 후 모델 입력에 결측·무한대가 남으면 명시적으로 중단"""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"모델 입력 컬럼이 없습니다: {sorted(missing)}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("전처리 후 모델 feature에 결측값 또는 무한대가 있습니다.")


def validate_targets(frame: pd.DataFrame, columns: list[str]) -> None:
    """관측 정답은 대치하지 않음, 회수율은 유한한 0~100%이어야 함"""
    for col in columns:
        if col not in frame:
            raise ValueError(f"목표변수 '{col}' 컬럼이 없습니다.")
        values = pd.to_numeric(frame[col], errors="coerce")
        invalid = ~np.isfinite(values) | ~values.between(0, 100)
        if invalid.any():
            batches = frame.loc[invalid, "배치ID"].tolist() if "배치ID" in frame else frame.index[invalid].tolist()
            raise ValueError(f"목표변수 '{col}'에 결측치, 무한대 또는 0~100% 범위 위반이 있습니다. 배치/행: {batches}")
