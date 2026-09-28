import pandas as pd
import pytest

from src.preprocessing import load_raw, clean_schema


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


def test_clean_schema_standardizes_mixed_date_formats():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003", "BP-2025-0004",
                   "BP-2025-0005", "BP-2025-0006"],
        "분석일자": ["2025-01-02", "2025/01/02", "03.01.2025", "20250104",
                    "Jan 05, 2025", "01/06/25"],
        "원료등급": ["A"] * 6,
    })

    result = clean_schema(df)

    assert result["분석일자"].tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-01-06"),
    ]


def test_clean_schema_returns_nat_for_unparseable_date():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001"],
        "분석일자": ["not-a-date"],
        "원료등급": ["A"],
    })

    result = clean_schema(df)

    assert pd.isna(result["분석일자"].iloc[0])


def test_clean_schema_normalizes_grade_variants():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003", "BP-2025-0004"],
        "분석일자": ["2025-01-02"] * 4,
        "원료등급": ["A급", " b ", "c", "a"],
    })

    result = clean_schema(df)

    assert result["원료등급"].tolist() == ["A", "B", "C", "A"]


def test_clean_schema_flags_duplicate_batch_ids():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0001", "BP-2025-0002"],
        "분석일자": ["2025-01-02"] * 3,
        "원료등급": ["A", "A", "B"],
    })

    result = clean_schema(df)

    assert result["is_duplicate_batch"].tolist() == [True, True, False]
