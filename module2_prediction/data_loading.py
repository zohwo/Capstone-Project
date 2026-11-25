import hashlib
import json
from pathlib import Path

import pandas as pd

from module2_prediction.config import CLEANED_DATA_PATH, PRE_IMPUTE_DATA_PATH, SELECTABLE_FEATURE_COLUMNS, TARGET_COLUMNS
from module1_preprocessing.transformation import RAW_INPUT_COLUMNS, validate_targets


def load_batch_data(path=CLEANED_DATA_PATH) -> pd.DataFrame:
    """Module 1의 배치 CSV를 읽고 배치ID를 문자열로 보존하며 공백 제거와 결측, 중복 검사를 수행"""

    df = pd.read_csv(path, dtype={"배치ID": "string"})

    if "배치ID" not in df.columns:
        raise ValueError(
            "`배치ID` 컬럼이 없습니다 — "
            "module1_preprocessing의 산출물이 맞는지 확인하세요."
        )

    df["배치ID"] = df["배치ID"].str.strip()

    if (df["배치ID"] == "").any() or df["배치ID"].isna().any():
        raise ValueError("`배치ID`에 결측값 또는 빈 문자열이 있습니다.")

    duplicated = df.loc[df["배치ID"].duplicated(keep=False), "배치ID"].unique().tolist()
    if duplicated:
        raise ValueError(
            f"`배치ID`가 중복됩니다 — 1행=1배치를 가정하는데, 같은 배치가 서로 다른 "
            f"split(train/validation/test)에 걸쳐 들어가면 평가 성능이 과대추정될 "
            f"수 있습니다: {duplicated}"
        )

    return df


def validate_split_column(df: pd.DataFrame) -> None:
    """split 컬럼의 존재, 결측 여부와 train/validation/test 값 구성을 검사"""

    if "split" not in df.columns:
        raise ValueError(
            "`split` 컬럼이 없습니다 — "
            "module1_preprocessing의 산출물이 맞는지 확인하세요."
        )

    if df["split"].isna().any():
        raise ValueError("`split` 컬럼에 결측값이 있습니다.")

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(df["split"].unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"`split` 컬럼 값이 예상과 다릅니다.\n"
            f"예상: {expected_splits}\n"
            f"실제: {actual_splits}"
        )


def split_by_column(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split 값을 검증하고 라벨별 데이터의 인덱스를 초기화하여 반환"""

    validate_split_column(df)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


def load_modeling_inputs(data_path=PRE_IMPUTE_DATA_PATH, selection_path=None):
    """Module 1의 정제, 분할 결과와 피처 선택 설정을 로드하고 유효성을 검증"""
    data_path = Path(data_path)
    selection_path = Path(selection_path) if selection_path else data_path.parent / "feature_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("feature_selection.json의 최상위 설정은 객체(dict)여야 합니다.")
    if (selection.get("schema_version") != 2
            or selection.get("preprocessing_stage") != "before_imputation"
            or selection.get("primary_metric") not in {"mae", "rmse"}):
        raise ValueError("선택 결과의 버전, 전처리 단계, 주 지표가 올바르지 않습니다. 현재 피처 구성으로 선택 결과를 재생성하세요.")
    if (selection.get("data_file") != data_path.name or
            selection.get("data_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest()):
        raise ValueError("데이터와 선택 결과가 일치하지 않습니다. 모듈 1을 다시 실행하세요.")
    df = load_batch_data(data_path)
    validate_split_column(df)
    missing = set(RAW_INPUT_COLUMNS + TARGET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"모델링 원본 컬럼이 없습니다: {sorted(missing)}")
    targets = selection.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("feature_selection.json의 targets는 객체(dict)여야 합니다.")
    if set(targets) != set(TARGET_COLUMNS):
        raise ValueError("타깃별 피처 선택 결과가 누락되었거나 예상과 다릅니다.")
    for target, entry in targets.items():
        if not isinstance(entry, dict):
            raise ValueError(f"feature_selection.json의 targets.{target}은 객체(dict)여야 합니다.")
        features = entry.get("features")
        if (not isinstance(features, list) or not features
                or any(not isinstance(f, str) for f in features)
                or len(set(features)) != len(features)
                or not set(features).issubset(set(SELECTABLE_FEATURE_COLUMNS))):
            raise ValueError(f"{target}의 선택 피처가 올바르지 않습니다.")
    cv = selection.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("feature_selection.json의 cv는 객체(dict)여야 합니다.")
    if (not isinstance(cv.get("n_splits"), int) or cv["n_splits"] < 2
            or cv.get("shuffle") is not True
            or not isinstance(cv.get("random_state"), int)
            or cv.get("stratify_column") != "원료등급"):
        raise ValueError("선택 결과의 CV 설정이 올바르지 않습니다.")
    validate_targets(df, TARGET_COLUMNS)
    df["분석일자"] = pd.to_datetime(df["분석일자"], errors="coerce")
    return df, selection


def main():
    """모델링 데이터를 로드하고 전체 및 split별 행 수를 출력"""
    df, _ = load_modeling_inputs()

    train_df, val_df, test_df = split_by_column(df)

    print("데이터 로드 및 분할 완료")
    print(f"전체 데이터: {len(df)}개")
    print(f"Train: {len(train_df)}개")
    print(f"Validation: {len(val_df)}개")
    print(f"Test: {len(test_df)}개")


if __name__ == "__main__":
    main()
