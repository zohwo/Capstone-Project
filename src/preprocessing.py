import re
from pathlib import Path

import pandas as pd

from src.config import RAW_SHEET_NAME


def load_raw(path: str | Path, sheet_name: str = RAW_SHEET_NAME) -> pd.DataFrame:
    """엑셀(xlsx/xls)의 지정된 시트 or CSV 파일을 읽어 DataFrame으로 반환한다."""

    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    if ext == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


_DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{2}\.\d{2}\.\d{4}$"), "%d.%m.%Y"),
    (re.compile(r"^\d{8}$"), "%Y%m%d"),
    (re.compile(r"^[A-Za-z]{3} \d{2}, \d{4}$"), "%b %d, %Y"),
    (re.compile(r"^\d{2}/\d{2}/\d{2}$"), "%m/%d/%y"),
]


def _parse_one_date(value) -> pd.Timestamp:
    """결측값 또는 지원하지 않는 날짜 형식은 NaT로 처리하고, 지원되는 날짜 형식은 Timestamp로 변환한다."""

    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for pattern, fmt in _DATE_PATTERNS:
        if pattern.match(text):
            try:
                return pd.to_datetime(text, format=fmt)
            except ValueError:
                return pd.NaT
    return pd.NaT


def clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 형식 통일, 원료등급 표기 정규화, 중복 배치ID 플래그를 적용한다.

    - 분석일자: 여러 날짜 표기를 실제 데이터 규칙에 맞게 통일해 파싱한다.
      점(.) 표기 날짜는 일-월-년 형식이므로 잘못 해석되지 않도록 형식을 직접 지정한다.
    - 원료등급: 앞뒤 공백, 영문 대소문자, '급' 표기를 통일한다.
    - 중복 배치 플래그: 동일한 배치ID가 2개 이상 존재하면 True로 표시한다.
    """

    df = df.copy()

    df["분석일자"] = df["분석일자"].apply(_parse_one_date)

    df["원료등급"] = (
        df["원료등급"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace("급", "", regex=False)
)

    df["is_duplicate_batch"] = df.duplicated(subset="배치ID", keep=False)

    return df


def resolve_duplicate_batches(df: pd.DataFrame) -> pd.DataFrame:
    """중복 배치ID를 확인해 완전 중복은 제거하고, 충돌 중복은 표시한다.

    - 완전 중복: 배치ID와 다른 컬럼 값까지 모두 같으면 첫 행만 남기고 제거한다.
    - 충돌 중복: 배치ID는 같지만 다른 컬럼 값이 다르면 삭제하지 않고
      `is_conflicting_duplicate=True`로 표시한다.
    - 중복 판단: `clean_schema`에서 생성한 `is_duplicate_batch` 플래그를 기준으로 처리한다.
    """

    df = df.copy()
    compare_cols = [c for c in df.columns if c != "is_duplicate_batch"]

    df["is_conflicting_duplicate"] = False
    rows_to_drop = []

    for _, group in df[df["is_duplicate_batch"]].groupby("배치ID"):
        is_exact_duplicate = (group[compare_cols].nunique(dropna=False) <= 1).all()

        if is_exact_duplicate:
            rows_to_drop.extend(group.index[1:])
        else:
            df.loc[group.index, "is_conflicting_duplicate"] = True

    return df.drop(index=rows_to_drop)
