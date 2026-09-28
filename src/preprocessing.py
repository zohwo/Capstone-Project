import re
from pathlib import Path

import pandas as pd

from src.config import RAW_SHEET_NAME


def load_raw(path: str | Path, sheet_name: str = RAW_SHEET_NAME) -> pd.DataFrame:
    """엑셀(xlsx/xls) 또는 CSV 파일을 읽어 DataFrame으로 반환한다."""
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

    분석일자는 형식별 정규식 매칭 후 명시적 strftime 포맷으로 파싱한다.
    pd.to_datetime(..., format="mixed")를 쓰지 않는 이유는 "03.01.2025" 같은
    점(.) 구분 표기를 월-일 순으로 오인해 실제 1월 3일을 3월 1일로 잘못
    파싱하기 때문이다 (이 데이터의 점 표기는 일-월-년 규칙).
    """
    df = df.copy()

    df["분석일자"] = df["분석일자"].apply(_parse_one_date)

    df["원료등급"] = (
        df["원료등급"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("급", "", regex=False)
    )

    df["is_duplicate_batch"] = df.duplicated(subset="배치ID", keep=False)

    return df
