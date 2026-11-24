"""모듈 1·2에서 사용하는 학습 데이터 기준 전처리와 입력 검증을 수행"""
import numpy as np
import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS
from module1_preprocessing.features import engineer_features
from module1_preprocessing.imputation import impute_missing_from_reference

RAW_INPUT_COLUMNS = COMPOSITION_COLUMNS + PROCESS_COLUMNS + [
    "원료등급", "원료투입량_kg", "황산투입량_L", "분석일자",
]


def prepare_split(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """참조 데이터 기준으로 대상 데이터의 결측값을 대치하고 분석일자 변환과 파생변수 생성을 수행"""
    for name, frame in (("reference", reference_df), ("target", target_df)):
        missing = set(RAW_INPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 데이터에 전처리 원본 컬럼이 없습니다: {sorted(missing)}")
    out = impute_missing_from_reference(reference_df, target_df)
    out["분석일자"] = pd.to_datetime(out["분석일자"], errors="coerce")
    return engineer_features(out)


def validate_model_features(frame: pd.DataFrame, columns: list[str]) -> None:
    """모델 입력 컬럼의 존재 여부와 결측·무한대 값을 검사"""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"모델 입력 컬럼이 없습니다: {sorted(missing)}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("전처리 후 모델 feature에 결측값 또는 무한대가 있습니다.")


def validate_targets(frame: pd.DataFrame, columns: list[str]) -> None:
    """타깃 컬럼의 존재 여부와 결측·무한대·0~100% 범위 위반을 검사"""
    for col in columns:
        if col not in frame:
            raise ValueError(f"목표변수 '{col}' 컬럼이 없습니다.")
        values = pd.to_numeric(frame[col], errors="coerce")
        invalid = ~np.isfinite(values) | ~values.between(0, 100)
        if invalid.any():
            batches = frame.loc[invalid, "배치ID"].tolist() if "배치ID" in frame else frame.index[invalid].tolist()
            raise ValueError(f"목표변수 '{col}'에 결측치, 무한대 또는 0~100% 범위 위반이 있습니다. 배치/행: {batches}")
