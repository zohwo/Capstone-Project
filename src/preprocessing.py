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
