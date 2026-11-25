import pandas as pd
import pytest

from module1_preprocessing.preprocessing import load_raw, clean_schema, resolve_duplicate_batches
from module1_preprocessing.config import NORMAL_RANGES, TARGET_COLUMNS


def _complete_inputs(df):
    """기존 개별 기능 테스트에 필수 숫자 입력을 추가하는 합성 데이터 보조 함수"""
    df = df.copy()
    for col, (low, high) in NORMAL_RANGES.items():
        if col not in df:
            df[col] = (low + high) / 2
    for col in TARGET_COLUMNS:
        if col not in df:
            df[col] = 90.0
    return df


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

    result = clean_schema(_complete_inputs(df))

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

    result = clean_schema(_complete_inputs(df))

    assert pd.isna(result["분석일자"].iloc[0])


def test_clean_schema_normalizes_grade_variants():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003", "BP-2025-0004"],
        "분석일자": ["2025-01-02"] * 4,
        "원료등급": ["A급", " b ", "c", "a"],
    })

    result = clean_schema(_complete_inputs(df))

    assert result["원료등급"].tolist() == ["A", "B", "C", "A"]


def test_clean_schema_flags_duplicate_batch_ids():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0001", "BP-2025-0002"],
        "분석일자": ["2025-01-02"] * 3,
        "원료등급": ["A", "A", "B"],
    })

    result = clean_schema(_complete_inputs(df))

    assert result["is_duplicate_batch"].tolist() == [True, True, False]


def test_resolve_duplicate_batches_drops_exact_duplicates_keeping_first():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0001", "BP-2025-0002"],
        "원료등급": ["A", "A", "B"],
        "Ni_함량_pct": [10.0, 10.0, 20.0],
        "is_duplicate_batch": [True, True, False],
    })

    result = resolve_duplicate_batches(df)

    assert result["배치ID"].tolist() == ["BP-2025-0001", "BP-2025-0002"]
    assert result["is_conflicting_duplicate"].tolist() == [False, False]


def test_resolve_duplicate_batches_keeps_conflicting_duplicates_and_flags_them():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0001", "BP-2025-0002"],
        "원료등급": ["A", "B", "C"],
        "Ni_함량_pct": [10.0, 15.0, 20.0],
        "is_duplicate_batch": [True, True, False],
    })

    result = resolve_duplicate_batches(df)

    assert len(result) == 3
    assert result["is_conflicting_duplicate"].tolist() == [True, True, False]


def test_resolve_duplicate_batches_treats_matching_nan_as_equal():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0001"],
        "원료등급": ["A", "A"],
        "Ni_함량_pct": [float("nan"), float("nan")],
        "is_duplicate_batch": [True, True],
    })

    result = resolve_duplicate_batches(df)

    assert len(result) == 1
    assert result["is_conflicting_duplicate"].tolist() == [False]


def test_excel_native_dates_survive_schema_cleaning(tmp_path):
    path = tmp_path / "dates.xlsx"
    expected = pd.Timestamp("2025-01-02")
    pd.DataFrame({"배치ID":["B1"], "분석일자":[expected], "원료등급":["A"]}).to_excel(
        path, sheet_name="원료분석_원본데이터", index=False)
    assert clean_schema(_complete_inputs(load_raw(path))).loc[0, "분석일자"] == expected


def _valid_input():
    return _complete_inputs(pd.DataFrame({
        '배치ID': ['B001', 'B002'], '분석일자': ['2025-01-02', '2025-01-03'],
        '원료등급': ['A', 'B'],
    }))


def test_schema_trims_columns_ids_and_grade_suffix_whitespace():
    df = _valid_input().rename(columns={'배치ID': ' 배치ID ', 'Ni_함량_pct': 'Ni_함량_pct '})
    df[' 배치ID '] = [' B001 ', 'B001']
    df['원료등급'] = [' a 급 ', 'B 급']
    out = clean_schema(df)
    assert out['배치ID'].tolist() == ['B001', 'B001']
    assert out['원료등급'].tolist() == ['A', 'B']
    assert out['is_duplicate_batch'].all()
    assert 'Ni_함량_pct' in out
    assert ' 배치ID ' in df  # 입력 원본 변경 없음


def test_schema_reports_all_missing_columns():
    df = _valid_input().drop(columns=['Ni_함량_pct', 'Co_회수율_pct'])
    with pytest.raises(ValueError, match='필수 컬럼') as error:
        clean_schema(df)
    assert 'Ni_함량_pct' in str(error.value)
    assert 'Co_회수율_pct' in str(error.value)


def test_schema_rejects_columns_colliding_after_trim():
    df = _valid_input()
    df[' Ni_함량_pct '] = 20
    with pytest.raises(ValueError, match='중복된 컬럼명'):
        clean_schema(df)


@pytest.mark.parametrize('value', [None, '', '  '])
def test_schema_rejects_missing_batch_id(value):
    df = _valid_input()
    df.loc[1, '배치ID'] = value
    with pytest.raises(ValueError, match=r'배치ID.*\[1\]'):
        clean_schema(df)


@pytest.mark.parametrize('value', [None, '', '  ', 'D', 'AB'])
def test_schema_rejects_missing_or_unknown_grade(value):
    df = _valid_input()
    df.loc[1, '원료등급'] = value
    with pytest.raises(ValueError, match=r'원료등급.*\[1\]'):
        clean_schema(df)


def test_schema_converts_numeric_strings_and_preserves_missing():
    df = _valid_input()
    df['Ni_함량_pct'] = [' 20.5 ', None]
    df['Co_함량_pct'] = ['', '   ']
    df['Ni_회수율_pct'] = ['90', None]
    out = clean_schema(df)
    assert out.loc[0, 'Ni_함량_pct'] == 20.5
    assert pd.isna(out.loc[1, 'Ni_함량_pct'])
    assert out['Co_함량_pct'].isna().all()
    assert out.loc[0, 'Ni_회수율_pct'] == 90
    assert pd.isna(out.loc[1, 'Ni_회수율_pct'])  # 타깃 대치 없음


def test_schema_reports_bad_numeric_column_and_row():
    df = _valid_input()
    df['Ni_함량_pct'] = ['20', '측정실패']
    with pytest.raises(ValueError, match=r'Ni_함량_pct.*\[1\].*측정실패'):
        clean_schema(df)


def test_schema_distinguishes_invalid_date_from_missing_date():
    df = pd.concat([_valid_input()] * 3, ignore_index=True)
    df['분석일자'] = ['not-a-date', None, '', '  ', '2025-02-30', '2025/01/02']
    out = clean_schema(df)
    assert out['analysis_date_parse_failed'].tolist() == [True, False, False, False, True, False]
    assert out['분석일자'].isna().tolist() == [True, True, True, True, True, False]


def test_pipeline_checks_schema_before_ranges_and_stops_on_conflict(tmp_path):
    from module1_preprocessing.pipeline import run
    df = _valid_input().astype({c: 'string' for c in NORMAL_RANGES})
    df['배치ID'] = [' B001 ', 'B001']
    df['Ni_함량_pct'] = ['20', '25']
    df.rename(columns={'원료등급': ' 원료등급 '}, inplace=True)
    path = tmp_path / 'input.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='충돌 중복'):
        run(path, tmp_path / 'out.csv')
    assert not (tmp_path / 'out.csv').exists()


def test_notebook_duplicate_cell_stops_on_conflict():
    import json
    from pathlib import Path
    nb = json.loads((Path(__file__).resolve().parents[2] / 'notebooks/01_eda_and_validation.ipynb').read_text())
    source = next(''.join(c['source']) for c in nb['cells']
                  if ''.join(c['source']).startswith('clean_df = resolve_duplicate_batches(clean_df)'))
    df = _valid_input()
    df['배치ID'] = ['B001', 'B001']
    with pytest.raises(ValueError, match='충돌 중복'):
        exec(source, {'clean_df': clean_schema(df), 'resolve_duplicate_batches': resolve_duplicate_batches})
