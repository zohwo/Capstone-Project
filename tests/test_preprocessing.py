import pandas as pd
import pytest

from src.preprocessing import load_raw


def test_load_raw_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"배치ID": ["BP-2025-0001"], "Ni_함량_pct": [10.0]}).to_csv(csv_path, index=False)

    df = load_raw(csv_path)

    assert list(df.columns) == ["배치ID", "Ni_함량_pct"]
    assert df.shape == (1, 2)


def test_load_raw_xlsx_uses_correct_sheet(tmp_path):
    xlsx_path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"배치ID": ["BP-2025-0001"]}).to_excel(
            writer, sheet_name="원료분석_원본데이터", index=False
        )
        pd.DataFrame({"컬럼명": ["배치ID"]}).to_excel(
            writer, sheet_name="데이터사전", index=False
        )

    df = load_raw(xlsx_path)

    assert list(df.columns) == ["배치ID"]
    assert df.shape == (1, 1)


def test_load_raw_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not supported")

    with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
        load_raw(bad_path)
