# 블랙파우더 성분 데이터 전처리 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 원본 랩실 분석 데이터(날짜 형식 6종 혼재, 결측치, 이상치, 범주 표기 불일치)를 도메인 지식 기반 규칙으로 정제하고, 회수율 예측에 유효한 파생변수를 추가해 `data/processed/cleaned_features.csv`로 저장하는 재사용 가능한 파이프라인을 구축한다.

**Architecture:** `src/` 아래 5개 순수 함수 모듈(`preprocessing.py`, `imputation.py`, `features.py`, `validation.py`, `pipeline.py`)로 구성된 파이프라인. 각 함수는 DataFrame을 입력받아 DataFrame(또는 dict/Path)을 반환하는 순수 함수라 노트북과 CLI 양쪽에서 동일하게 재사용 가능. `pipeline.py`가 전체 단계를 순서대로 호출하는 엔트리포인트.

**Tech Stack:** pandas, numpy, scikit-learn(LinearRegression, RandomForestRegressor, KFold), matplotlib, seaborn, openpyxl, pytest, jupyter/nbformat/nbconvert

참고 설계 문서: [`docs/superpowers/specs/2026-09-28-blackpowder-preprocessing-design.md`](../specs/2026-09-28-blackpowder-preprocessing-design.md)

## Global Constraints

- 원본 데이터 경로: `data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx`, 메인 시트명 `원료분석_원본데이터` (시트 `데이터사전`은 로드하지 않음)
- 모든 DataFrame 변환 함수는 순수 함수로 작성한다 — 입력 `df`를 직접 변형하지 않고 `df.copy()` 후 반환한다
- 정상 범위·컬럼 그룹 정의는 `src/config.py` 한 곳에서만 관리한다 (DRY) — 다른 모듈에서 하드코딩 금지
- `data/raw/`, `data/processed/`, `reports/figures/`는 `.gitignore`에 이미 등록되어 있으므로 커밋 대상에서 제외된 상태를 유지한다
- 모든 테스트는 `tests/` 아래 pytest로 작성하고, 테스트 실행은 프로젝트 루트에서 `pytest`로 한다 (루트의 `pytest.ini`가 `pythonpath = .`를 설정)
- `분석일자`는 절대 `pd.to_datetime(..., format="mixed")`로 일괄 처리하지 않는다 — 점(`.`) 구분 형식을 월-일로 오인해 잘못 파싱하므로, 형식별 정규식 매칭 후 명시적 `strftime` 포맷으로 파싱한다 (Task 3 참고)
- 각 태스크 완료 시 해당 태스크에서 변경한 파일만 `git add`하고 커밋한다 (전체 `git add -A` 금지)

---

## Task 1: 프로젝트 설정 (config, requirements, pytest)

**Files:**
- Create: `src/config.py`
- Create: `requirements.txt` (덮어쓰기, 현재 빈 파일)
- Create: `pytest.ini`

**Interfaces:**
- Produces: `NORMAL_RANGES: dict[str, tuple[float, float]]`, `COMPOSITION_COLUMNS: list[str]`, `PROCESS_COLUMNS: list[str]`, `TARGET_COLUMNS: list[str]`, `RAW_SHEET_NAME: str` — 이후 모든 태스크가 이 상수들을 import해서 사용

- [ ] **Step 1: `src/config.py` 작성**

```python
"""원료분석 데이터의 정상 범위 및 컬럼 그룹 정의 (데이터사전 기준, 단일 소스)."""

NORMAL_RANGES = {
    "Ni_함량_pct": (8, 30),
    "Co_함량_pct": (2, 15),
    "Li_함량_pct": (2, 6),
    "Mn_함량_pct": (5, 16),
    "Cu_불순물_pct": (0.3, 6),
    "Fe_불순물_pct": (0.2, 6),
    "Al_불순물_pct": (0.5, 7),
    "원료투입량_kg": (80, 200),
    "황산투입량_L": (400, 1500),
    "침출온도_C": (40, 95),
    "침출시간_hr": (1, 8),
}

COMPOSITION_COLUMNS = [
    "Ni_함량_pct",
    "Co_함량_pct",
    "Li_함량_pct",
    "Mn_함량_pct",
    "Cu_불순물_pct",
    "Fe_불순물_pct",
    "Al_불순물_pct",
]

PROCESS_COLUMNS = ["침출온도_C", "침출시간_hr"]

TARGET_COLUMNS = ["Ni_회수율_pct", "Co_회수율_pct"]

RAW_SHEET_NAME = "원료분석_원본데이터"
```

- [ ] **Step 2: `requirements.txt` 작성**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
pytest>=7.4
jupyter>=1.0
nbformat>=5.9
nbconvert>=7.0
```

- [ ] **Step 3: `pytest.ini` 작성**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 4: 의존성 설치 및 확인**

Run: `pip install -r requirements.txt`
Expected: 에러 없이 설치 완료

- [ ] **Step 5: config import가 되는지 확인**

Run: `python -c "from src.config import NORMAL_RANGES; print(len(NORMAL_RANGES))"`
Expected: `11`

- [ ] **Step 6: 커밋**

```bash
git add src/config.py requirements.txt pytest.ini
git commit -m "chore: add shared config and project setup for preprocessing pipeline"
```

---

## Task 2: `load_raw` — 파일 로더

**Files:**
- Modify: `src/preprocessing.py` (기존 `read_data_file`/`convert_data`/`__main__` 스텁을 전면 교체)
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `src.config.RAW_SHEET_NAME`
- Produces: `load_raw(path: str | Path, sheet_name: str = RAW_SHEET_NAME) -> pd.DataFrame`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL (`ImportError: cannot import name 'load_raw'`)

- [ ] **Step 3: 최소 구현 작성**

`src/preprocessing.py` (파일 전체를 아래 내용으로 교체):

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_preprocessing.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add load_raw for csv/xlsx loading with sheet selection"
```

---

## Task 3: `clean_schema` — 날짜 파싱·범주 정규화·중복 플래그

**Files:**
- Modify: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `clean_schema(df: pd.DataFrame) -> pd.DataFrame` — `분석일자`는 datetime64로 통일, `원료등급`은 `A`/`B`/`C`로 정규화, `is_duplicate_batch: bool` 컬럼 추가

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_preprocessing.py`에 추가:

```python
from src.preprocessing import clean_schema


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_preprocessing.py -v -k clean_schema`
Expected: FAIL (`ImportError: cannot import name 'clean_schema'`)

- [ ] **Step 3: 구현 작성**

`src/preprocessing.py`에 추가 (파일 상단에 `import re` 추가):

```python
import re

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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_preprocessing.py -v`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add clean_schema for date parsing, grade normalization, and dup detection"
```

---

## Task 4: `flag_outliers` — 정상범위 이탈 값을 결측으로 변환

**Files:**
- Create: `src/imputation.py`
- Test: `tests/test_imputation.py`

**Interfaces:**
- Consumes: `src.config.NORMAL_RANGES`
- Produces: `flag_outliers(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> pd.DataFrame`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_imputation.py`:

```python
import numpy as np
import pandas as pd

from src.imputation import flag_outliers


def test_flag_outliers_converts_out_of_range_values_to_nan():
    df = pd.DataFrame({
        "Ni_함량_pct": [10.0, 999.0, 25.0],
        "침출온도_C": [60.0, -5.0, 70.0],
    })

    result = flag_outliers(df, ranges={"Ni_함량_pct": (8, 30), "침출온도_C": (40, 95)})

    assert result["Ni_함량_pct"].iloc[0] == 10.0
    assert np.isnan(result["Ni_함량_pct"].iloc[1])
    assert result["Ni_함량_pct"].iloc[2] == 25.0
    assert result["침출온도_C"].iloc[0] == 60.0
    assert np.isnan(result["침출온도_C"].iloc[1])
    assert result["침출온도_C"].iloc[2] == 70.0


def test_flag_outliers_keeps_values_within_range_unchanged():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 30.0]})

    result = flag_outliers(df, ranges={"Ni_함량_pct": (8, 30)})

    assert result["Ni_함량_pct"].tolist() == [10.0, 20.0, 30.0]


def test_flag_outliers_ignores_columns_not_in_dataframe():
    df = pd.DataFrame({"Ni_함량_pct": [10.0]})

    result = flag_outliers(df, ranges={"존재하지않는컬럼": (0, 1)})

    assert result["Ni_함량_pct"].tolist() == [10.0]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_imputation.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.imputation'`)

- [ ] **Step 3: 구현 작성**

`src/imputation.py`:

```python
import numpy as np
import pandas as pd

from src.config import NORMAL_RANGES


def flag_outliers(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> pd.DataFrame:
    """데이터사전 정상 범위를 벗어난 값을 NaN으로 치환한다 (이상치→결측 변환)."""
    df = df.copy()

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < low) | (df[col] > high)
        df.loc[out_of_range, col] = np.nan

    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_imputation.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/imputation.py tests/test_imputation.py
git commit -m "feat: add flag_outliers to convert out-of-range values to NaN"
```

---

## Task 5: `impute_missing` — 그룹별/중앙값 대치 (조성·공정조건)

**Files:**
- Modify: `src/imputation.py`
- Test: `tests/test_imputation.py`

**Interfaces:**
- Consumes: `src.config.COMPOSITION_COLUMNS`, `src.config.PROCESS_COLUMNS`
- Produces: `impute_missing(df: pd.DataFrame) -> pd.DataFrame` (이번 태스크에서는 조성 그룹별 중앙값 + 공정조건 중앙값만 처리; Task 6/7에서 황산투입량 회귀·분석일자 보간을 추가로 연결)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_imputation.py`에 추가:

```python
from src.imputation import impute_missing


def test_impute_missing_fills_composition_with_grade_group_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A", "C", "C", "C"],
        "Ni_함량_pct": [20.0, None, 24.0, 10.0, None, 12.0],
        "Co_함량_pct": [5.0] * 6,
        "Li_함량_pct": [3.0] * 6,
        "Mn_함량_pct": [9.0] * 6,
        "Cu_불순물_pct": [1.0] * 6,
        "Fe_불순물_pct": [1.0] * 6,
        "Al_불순물_pct": [1.0] * 6,
        "침출온도_C": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0],
        "침출시간_hr": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    })

    result = impute_missing(df)

    # A그룹 Ni 중앙값: median(20.0, 24.0) = 22.0 / C그룹: median(10.0, 12.0) = 11.0
    assert result["Ni_함량_pct"].tolist() == [20.0, 22.0, 24.0, 10.0, 11.0, 12.0]


def test_impute_missing_fills_process_columns_with_overall_median():
    df = pd.DataFrame({
        "원료등급": ["A", "A", "A"],
        "Ni_함량_pct": [20.0] * 3,
        "Co_함량_pct": [5.0] * 3,
        "Li_함량_pct": [3.0] * 3,
        "Mn_함량_pct": [9.0] * 3,
        "Cu_불순물_pct": [1.0] * 3,
        "Fe_불순물_pct": [1.0] * 3,
        "Al_불순물_pct": [1.0] * 3,
        "침출온도_C": [60.0, None, 80.0],
        "침출시간_hr": [3.0, 4.0, None],
    })

    result = impute_missing(df)

    assert result["침출온도_C"].tolist() == [60.0, 70.0, 80.0]
    assert result["침출시간_hr"].tolist() == [3.0, 4.0, 3.5]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_imputation.py -v -k impute_missing`
Expected: FAIL (`ImportError: cannot import name 'impute_missing'`)

- [ ] **Step 3: 구현 작성**

`src/imputation.py`에 추가:

```python
from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """조성 변수는 원료등급 그룹별 중앙값, 공정조건은 전체 중앙값으로 대치한다."""
    df = df.copy()

    for col in COMPOSITION_COLUMNS:
        df[col] = df.groupby("원료등급")[col].transform(lambda s: s.fillna(s.median()))

    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_imputation.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/imputation.py tests/test_imputation.py
git commit -m "feat: add impute_missing for grade-group and overall median imputation"
```

---

## Task 6: `impute_acid_volume` — 회귀 기반 대치 (황산투입량_L)

**Files:**
- Modify: `src/imputation.py`
- Test: `tests/test_imputation.py`

**Interfaces:**
- Produces: `impute_acid_volume(df: pd.DataFrame) -> pd.DataFrame`
- `impute_missing`을 수정해 내부에서 `impute_acid_volume`을 호출하도록 연결

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_imputation.py`에 추가:

```python
from src.imputation import impute_acid_volume


def test_impute_acid_volume_predicts_missing_from_ore_mass():
    # 정확히 acid = 7 * ore 관계를 갖는 학습 데이터 + 결측 1건
    df = pd.DataFrame({
        "원료투입량_kg": [100.0, 120.0, 140.0, 160.0, 180.0],
        "황산투입량_L": [700.0, 840.0, 980.0, None, 1260.0],
    })

    result = impute_acid_volume(df)

    assert result["황산투입량_L"].isna().sum() == 0
    assert result["황산투입량_L"].iloc[3] == pytest.approx(1120.0, rel=1e-6)


def test_impute_acid_volume_leaves_complete_column_unchanged():
    df = pd.DataFrame({
        "원료투입량_kg": [100.0, 120.0],
        "황산투입량_L": [700.0, 840.0],
    })

    result = impute_acid_volume(df)

    assert result["황산투입량_L"].tolist() == [700.0, 840.0]
```

`tests/test_imputation.py` 상단에 `import pytest` 추가 (아직 없다면).

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_imputation.py -v -k impute_acid_volume`
Expected: FAIL (`ImportError: cannot import name 'impute_acid_volume'`)

- [ ] **Step 3: 구현 작성**

`src/imputation.py` 상단에 `from sklearn.linear_model import LinearRegression` 추가, 함수 추가:

```python
from sklearn.linear_model import LinearRegression


def impute_acid_volume(df: pd.DataFrame) -> pd.DataFrame:
    """황산투입량_L 결측을 원료투입량_kg 기반 선형회귀(산-광석비)로 대치한다."""
    df = df.copy()

    missing_mask = df["황산투입량_L"].isna()
    if not missing_mask.any():
        return df

    train = df.loc[~missing_mask, ["원료투입량_kg", "황산투입량_L"]].dropna()
    model = LinearRegression().fit(train[["원료투입량_kg"]], train["황산투입량_L"])

    df.loc[missing_mask, "황산투입량_L"] = model.predict(
        df.loc[missing_mask, ["원료투입량_kg"]]
    )

    return df
```

`impute_missing` 함수 마지막에 호출 추가:

```python
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COMPOSITION_COLUMNS:
        df[col] = df.groupby("원료등급")[col].transform(lambda s: s.fillna(s.median()))

    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    df = impute_acid_volume(df)

    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_imputation.py -v`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add src/imputation.py tests/test_imputation.py
git commit -m "feat: add regression-based imputation for acid volume"
```

---

## Task 7: `interpolate_analysis_date` — 배치ID 순서 기반 날짜 보간

**Files:**
- Modify: `src/imputation.py`
- Test: `tests/test_imputation.py`

**Interfaces:**
- Produces: `interpolate_analysis_date(df: pd.DataFrame) -> pd.DataFrame`
- `impute_missing`을 수정해 내부에서 `interpolate_analysis_date`도 호출하도록 연결 (최종 형태)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_imputation.py`에 추가:

```python
from src.imputation import interpolate_analysis_date


def test_interpolate_analysis_date_fills_gap_using_batch_order():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003", "BP-2025-0004"],
        "분석일자": [
            pd.Timestamp("2025-01-01"),
            pd.NaT,
            pd.NaT,
            pd.Timestamp("2025-01-04"),
        ],
    })

    result = interpolate_analysis_date(df)

    assert result["분석일자"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    ]


def test_interpolate_analysis_date_extrapolates_boundary_gap():
    df = pd.DataFrame({
        "배치ID": ["BP-2025-0001", "BP-2025-0002", "BP-2025-0003"],
        "분석일자": [pd.NaT, pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
    })

    result = interpolate_analysis_date(df)

    assert result["분석일자"].isna().sum() == 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_imputation.py -v -k interpolate_analysis_date`
Expected: FAIL (`ImportError: cannot import name 'interpolate_analysis_date'`)

- [ ] **Step 3: 구현 작성**

`src/imputation.py`에 추가:

```python
def interpolate_analysis_date(df: pd.DataFrame) -> pd.DataFrame:
    """배치ID의 순번을 시간 순서 근사값으로 사용해 분석일자 결측을 보간한다."""
    df = df.copy()

    batch_seq = df["배치ID"].str.extract(r"(\d+)$")[0].astype(int)
    ordered_index = batch_seq.sort_values().index

    dates = df.loc[ordered_index, "분석일자"]
    numeric = dates.to_numpy().astype("int64").astype("float64")
    numeric[dates.isna().to_numpy()] = float("nan")

    interpolated = pd.Series(numeric, index=ordered_index).interpolate(
        limit_direction="both"
    )
    df.loc[ordered_index, "분석일자"] = pd.to_datetime(
        interpolated.round().astype("int64")
    )

    return df
```

`impute_missing` 함수 마지막에 호출 추가 (최종 형태):

```python
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COMPOSITION_COLUMNS:
        df[col] = df.groupby("원료등급")[col].transform(lambda s: s.fillna(s.median()))

    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    df = impute_acid_volume(df)
    df = interpolate_analysis_date(df)

    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_imputation.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add src/imputation.py tests/test_imputation.py
git commit -m "feat: add batch-order date interpolation and wire into impute_missing"
```

---

## Task 8: `engineer_features` — 파생변수 5종

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `engineer_features(df: pd.DataFrame) -> pd.DataFrame` — 컬럼 `acid_to_ore_ratio`, `leach_intensity`, `total_impurity_pct`, `target_purity_ratio`, `analysis_month` 추가

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_features.py`:

```python
import pandas as pd
import pytest

from src.features import engineer_features


def test_engineer_features_computes_all_derived_columns():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    result = engineer_features(df)

    assert result["acid_to_ore_ratio"].iloc[0] == pytest.approx(7.0)
    assert result["leach_intensity"].iloc[0] == pytest.approx(240.0)
    assert result["total_impurity_pct"].iloc[0] == pytest.approx(3.0)
    assert result["target_purity_ratio"].iloc[0] == pytest.approx(25.0 / 28.0)
    assert result["analysis_month"].iloc[0] == 3


def test_engineer_features_does_not_mutate_input():
    df = pd.DataFrame({
        "황산투입량_L": [700.0],
        "원료투입량_kg": [100.0],
        "침출온도_C": [60.0],
        "침출시간_hr": [4.0],
        "Cu_불순물_pct": [1.0],
        "Fe_불순물_pct": [1.0],
        "Al_불순물_pct": [1.0],
        "Ni_함량_pct": [20.0],
        "Co_함량_pct": [5.0],
        "분석일자": [pd.Timestamp("2025-03-15")],
    })

    engineer_features(df)

    assert "acid_to_ore_ratio" not in df.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.features'`)

- [ ] **Step 3: 구현 작성**

`src/features.py`:

```python
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """침출 공정 도메인 지식 기반 파생변수 5종을 추가한다.

    - acid_to_ore_ratio: 산-광석비 (침출 화학양론 지표)
    - leach_intensity: 침출강도 = 온도 x 시간 (반응속도론적 진행도 근사)
    - total_impurity_pct: 총불순물 (경쟁 반응 유발 요인)
    - target_purity_ratio: 목표금속(Ni+Co) 순도비
    - analysis_month: 계절성/설비 드리프트 탐색용
    """
    df = df.copy()

    df["acid_to_ore_ratio"] = df["황산투입량_L"] / df["원료투입량_kg"]
    df["leach_intensity"] = df["침출온도_C"] * df["침출시간_hr"]
    df["total_impurity_pct"] = (
        df["Cu_불순물_pct"] + df["Fe_불순물_pct"] + df["Al_불순물_pct"]
    )

    target_metals = df["Ni_함량_pct"] + df["Co_함량_pct"]
    df["target_purity_ratio"] = target_metals / (target_metals + df["total_impurity_pct"])

    df["analysis_month"] = df["분석일자"].dt.month

    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_features.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add engineer_features with 5 domain-derived variables"
```

---

## Task 9: `check_ranges` — 정합성 검증

**Files:**
- Create: `src/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `src.config.NORMAL_RANGES`
- Produces: `check_ranges(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> dict[str, list]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validation.py`:

```python
import pandas as pd

from src.validation import check_ranges


def test_check_ranges_reports_out_of_range_row_indices():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 999.0, 25.0]}, index=[0, 1, 2])

    violations = check_ranges(df, ranges={"Ni_함량_pct": (8, 30)})

    assert violations == {"Ni_함량_pct": [1]}


def test_check_ranges_returns_empty_dict_when_all_within_range():
    df = pd.DataFrame({"Ni_함량_pct": [10.0, 20.0, 25.0]})

    violations = check_ranges(df, ranges={"Ni_함량_pct": (8, 30)})

    assert violations == {}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.validation'`)

- [ ] **Step 3: 구현 작성**

`src/validation.py`:

```python
import pandas as pd

from src.config import NORMAL_RANGES


def check_ranges(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> dict:
    """정제 후 데이터가 데이터사전의 정상 범위를 만족하는지 검증한다.

    반환값: {컬럼명: [위반 행 인덱스, ...]} — 위반이 없으면 빈 dict
    """
    violations = {}

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            violations[col] = out_of_range.index.tolist()

    return violations
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_validation.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add check_ranges for post-cleaning data validation"
```

---

## Task 10: `pipeline.run` — 오케스트레이터 + 실데이터 통합 테스트

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `load_raw`, `clean_schema` (preprocessing.py); `flag_outliers`, `impute_missing` (imputation.py); `engineer_features` (features.py); `check_ranges` (validation.py)
- Produces: `run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

`tests/test_pipeline.py`:

```python
from pathlib import Path

import pandas as pd

from src.config import NORMAL_RANGES
from src.pipeline import run

RAW_PATH = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")


def test_run_produces_clean_dataframe_with_no_missing_in_key_columns(tmp_path):
    output_path = tmp_path / "cleaned_features.csv"

    result = run(RAW_PATH, output_path)

    assert output_path.exists()
    assert result.shape[0] == 424
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
    ]:
        assert col in result.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.pipeline'`)

- [ ] **Step 3: 구현 작성**

`src/pipeline.py`:

```python
from pathlib import Path

import pandas as pd

from src.preprocessing import load_raw, clean_schema
from src.imputation import flag_outliers, impute_missing
from src.features import engineer_features
from src.validation import check_ranges


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """원본 데이터를 로드해 정제·대치·특성공학·검증까지 수행하고 CSV로 저장한다."""
    df = load_raw(input_path)
    df = clean_schema(df)
    df = flag_outliers(df)
    df = impute_missing(df)
    df = engineer_features(df)

    violations = check_ranges(df)
    if violations:
        print(f"경고: 정상범위 위반 발견 - {violations}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    run(
        Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"),
        Path("data/processed/cleaned_features.csv"),
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: 3 passed

(실패 시 가장 흔한 원인: `impute_acid_volume`이 처리하기 전 `원료투입량_kg`에 남은 결측치, 또는 `원료등급` 그룹에 해당 등급 전체가 결측이라 그룹 중앙값도 NaN인 경우. 실데이터 확인 결과 두 경우 모두 해당 없음.)

- [ ] **Step 5: CLI로 실제 실행 확인**

Run: `python -m src.pipeline`
Expected: `data/processed/cleaned_features.csv` 생성, 콘솔에 경고 없음(정상범위 위반 0건)

- [ ] **Step 6: 커밋**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with end-to-end integration tests"
```

---

## Task 11: 검증 시각화 (`plot_missing_before_after`, `plot_distributions`, `plot_correlation_heatmap`)

**Files:**
- Modify: `src/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `plot_missing_before_after(before: pd.DataFrame, after: pd.DataFrame, output_dir: str | Path) -> Path`
  - `plot_distributions(df: pd.DataFrame, columns: list[str], output_dir: str | Path) -> Path`
  - `plot_correlation_heatmap(df: pd.DataFrame, feature_columns: list[str], target_columns: list[str], output_dir: str | Path) -> Path`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validation.py`에 추가:

```python
import numpy as np

from src.validation import plot_missing_before_after, plot_distributions, plot_correlation_heatmap


def test_plot_missing_before_after_creates_png(tmp_path):
    before = pd.DataFrame({"Ni_함량_pct": [1.0, None, 3.0]})
    after = pd.DataFrame({"Ni_함량_pct": [1.0, 2.0, 3.0]})

    output_path = plot_missing_before_after(before, after, tmp_path)

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_distributions_creates_png(tmp_path):
    df = pd.DataFrame({
        "Ni_함량_pct": np.random.default_rng(0).normal(20, 3, 30),
        "황산투입량_L": np.random.default_rng(1).normal(900, 100, 30),
    })

    output_path = plot_distributions(df, ["Ni_함량_pct", "황산투입량_L"], tmp_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_correlation_heatmap_creates_png(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, 30),
        "Co_함량_pct": rng.normal(8, 2, 30),
        "Ni_회수율_pct": rng.normal(85, 5, 30),
    })

    output_path = plot_correlation_heatmap(
        df, ["Ni_함량_pct", "Co_함량_pct"], ["Ni_회수율_pct"], tmp_path
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_validation.py -v -k plot`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: 구현 작성**

`src/validation.py` 상단에 추가:

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
```

함수 추가:

```python
def plot_missing_before_after(
    before: pd.DataFrame, after: pd.DataFrame, output_dir: str | Path
) -> Path:
    """컬럼별 결측 비율을 처리 전/후로 비교하는 막대그래프를 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_pct = before.isna().mean() * 100
    after_pct = after.isna().mean() * 100
    columns = [c for c in before_pct.index if c in after_pct.index]

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.8), 5))
    x = range(len(columns))
    ax.bar([i - 0.2 for i in x], before_pct[columns], width=0.4, label="처리 전")
    ax.bar([i + 0.2 for i in x], after_pct[columns], width=0.4, label="처리 후")
    ax.set_xticks(list(x))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_ylabel("결측 비율 (%)")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "missing_before_after.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_distributions(df: pd.DataFrame, columns: list, output_dir: str | Path) -> Path:
    """지정 컬럼들의 분포를 박스플롯으로 저장한다 (이상치 확인용)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        ax.boxplot(df[col].dropna())
        ax.set_title(col)
    fig.tight_layout()

    output_path = output_dir / "distributions.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(
    df: pd.DataFrame, feature_columns: list, target_columns: list, output_dir: str | Path
) -> Path:
    """피처(파생변수 포함) vs 목표변수 상관관계 히트맵을 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr = df[feature_columns + target_columns].corr().loc[feature_columns, target_columns]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_columns) * 0.4)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()

    output_path = output_dir / "correlation_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_validation.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add missing/distribution/correlation visualization for validation"
```

---

## Task 12: `run_ablation_test` — 파생변수·원료등급 성능 기여 검증

**Files:**
- Modify: `src/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces: `run_ablation_test(df: pd.DataFrame, feature_sets: dict[str, list[str]], target_columns: list[str], n_splits: int = 5, random_state: int = 42) -> dict[tuple[str, str], dict[str, float]]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_validation.py`에 추가:

```python
from src.validation import run_ablation_test


def test_run_ablation_test_returns_r2_and_rmse_per_case_and_target():
    rng = np.random.default_rng(42)
    n = 40
    x1 = rng.normal(20, 3, n)
    x2 = rng.normal(8, 2, n)
    noise = rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "target_1": x1 * 2 + x2 + noise,
    })

    feature_sets = {
        "base": ["feature_a"],
        "base_plus_b": ["feature_a", "feature_b"],
    }

    results = run_ablation_test(df, feature_sets, ["target_1"], n_splits=3, random_state=0)

    assert set(results.keys()) == {("base", "target_1"), ("base_plus_b", "target_1")}
    for metrics in results.values():
        assert "r2_mean" in metrics
        assert "rmse_mean" in metrics
        assert isinstance(metrics["r2_mean"], float)
        assert isinstance(metrics["rmse_mean"], float)


def test_run_ablation_test_encodes_categorical_features():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "grade": rng.choice(["A", "B", "C"], n),
        "value": rng.normal(10, 2, n),
        "target_1": rng.normal(50, 5, n),
    })

    feature_sets = {"with_grade": ["grade", "value"]}

    results = run_ablation_test(df, feature_sets, ["target_1"], n_splits=3, random_state=0)

    assert ("with_grade", "target_1") in results
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_validation.py -v -k ablation`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: 구현 작성**

`src/validation.py` 상단에 추가:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
```

함수 추가:

```python
def run_ablation_test(
    df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """피처 조합별로 교차검증 R²/RMSE를 계산해 파생변수·원료등급의 성능 기여를 정량 비교한다."""
    results = {}

    for case_name, feature_columns in feature_sets.items():
        X = pd.get_dummies(df[feature_columns])

        for target in target_columns:
            y = df[target]
            model = RandomForestRegressor(random_state=random_state)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_validate(
                model, X, y, cv=cv, scoring=["r2", "neg_root_mean_squared_error"]
            )
            results[(case_name, target)] = {
                "r2_mean": float(scores["test_r2"].mean()),
                "rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            }

    return results
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_validation.py -v`
Expected: 7 passed

- [ ] **Step 5: 전체 테스트 스위트 확인**

Run: `pytest -v`
Expected: 모든 테스트 통과 (Task 2~12 누적 테스트, 약 27개)

- [ ] **Step 6: 커밋**

```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add ablation test for derived feature and grade contribution validation"
```

---

## Task 13: EDA/검증 노트북 생성 및 심층 인터뷰 답변 정리

**Files:**
- Create: `scripts/build_notebook.py` (노트북 생성 스크립트, 재실행 가능하도록 보관)
- Create: `notebooks/01_eda_and_validation.ipynb` (스크립트 실행 결과물)

**Interfaces:**
- Consumes: `load_raw`, `clean_schema`, `flag_outliers`, `impute_missing`, `engineer_features`, `check_ranges`, `plot_missing_before_after`, `plot_distributions`, `plot_correlation_heatmap`, `run_ablation_test`, `COMPOSITION_COLUMNS`, `PROCESS_COLUMNS`, `TARGET_COLUMNS`

- [ ] **Step 1: 노트북 생성 스크립트 작성**

`scripts/build_notebook.py`:

```python
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 1: 블랙파우더 성분 데이터 전처리 - EDA 및 검증\n\n"
    "`src/pipeline.py`의 각 단계를 순서대로 호출해 원본 데이터를 정제하고, "
    "전/후 비교 시각화와 파생변수·원료등급 기여도(ablation test)를 확인한다."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "from pathlib import Path\n\n"
    "from src.preprocessing import load_raw, clean_schema\n"
    "from src.imputation import flag_outliers, impute_missing\n"
    "from src.features import engineer_features\n"
    "from src.validation import (\n"
    "    check_ranges,\n"
    "    plot_missing_before_after,\n"
    "    plot_distributions,\n"
    "    plot_correlation_heatmap,\n"
    "    run_ablation_test,\n"
    ")\n"
    "from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS\n\n"
    "RAW_PATH = Path('../data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx')\n"
    "FIGURES_DIR = Path('../reports/figures')"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 원본 로드 및 정제"))

cells.append(nbf.v4.new_code_cell(
    "raw_df = load_raw(RAW_PATH)\n"
    "clean_df = clean_schema(raw_df)\n"
    "outlier_flagged_df = flag_outliers(clean_df)\n"
    "imputed_df = impute_missing(outlier_flagged_df)\n"
    "final_df = engineer_features(imputed_df)\n"
    "final_df.head()"
))

cells.append(nbf.v4.new_markdown_cell("## 2. 결측치 처리 전/후 비교"))

cells.append(nbf.v4.new_code_cell(
    "plot_missing_before_after(outlier_flagged_df, imputed_df, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell("## 3. 이상치 제거 전/후 분포"))

cells.append(nbf.v4.new_code_cell(
    "plot_distributions(raw_df, ['Ni_함량_pct', '황산투입량_L', '침출온도_C'], FIGURES_DIR / 'before')\n"
    "plot_distributions(final_df, ['Ni_함량_pct', '황산투입량_L', '침출온도_C'], FIGURES_DIR / 'after')"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 상관관계 히트맵"))

cells.append(nbf.v4.new_code_cell(
    "feature_columns = COMPOSITION_COLUMNS + PROCESS_COLUMNS + [\n"
    "    'acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'target_purity_ratio'\n"
    "]\n"
    "plot_correlation_heatmap(final_df, feature_columns, TARGET_COLUMNS, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell("## 5. 정합성 검증"))

cells.append(nbf.v4.new_code_cell(
    "violations = check_ranges(final_df)\n"
    "print('정상범위 위반:', violations if violations else '없음')"
))

cells.append(nbf.v4.new_markdown_cell("## 6. 파생변수 및 원료등급 기여도 검증 (Ablation Test)"))

cells.append(nbf.v4.new_code_cell(
    "base_features = COMPOSITION_COLUMNS + PROCESS_COLUMNS\n"
    "derived_features = ['acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'target_purity_ratio']\n\n"
    "feature_sets = {\n"
    "    'base_only': base_features,\n"
    "    'base_plus_grade': base_features + ['원료등급'],\n"
    "    'base_plus_derived': base_features + derived_features,\n"
    "    'all_features': base_features + ['원료등급'] + derived_features,\n"
    "}\n\n"
    "results = run_ablation_test(final_df, feature_sets, TARGET_COLUMNS)\n"
    "pd.DataFrame(results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. 심층 인터뷰 답변\n\n"
    "**1) 실제 공정 데이터 품질 문제를 어떻게 식별·처리했는가?**\n\n"
    "데이터사전에 정의된 정상 범위를 벗어나는 값(`Ni_함량_pct=999.0`, `침출온도_C=-5.0`, "
    "`황산투입량_L=15000`)을 물리적으로 불가능한 이상치로 식별해 `flag_outliers`에서 NaN으로 "
    "치환했다. 이후 남은 랜덤 결측(조성 6~9%, 공정조건)은 변수 성격에 따라 원료등급별 중앙값, "
    "회귀 기반, 배치ID 순서 기반 보간으로 구분 처리했다 (설계 문서 §4).\n\n"
    "**2) 특성공학 과정에서 도메인 지식을 어떻게 활용했는가?**\n\n"
    "산-광석비, 침출강도(온도×시간), 총불순물, 목표금속 순도비는 모두 습식제련 침출 공정의 "
    "화학양론·반응속도론적 지식에서 도출했다 (설계 문서 §6). 위 ablation test로 파생변수·원료등급 "
    "포함 시 R²/RMSE가 실제로 개선되는지 정량 확인했다.\n\n"
    "**3) 데이터 불균형 문제가 있었다면 어떻게 해결했는가?**\n\n"
    "`원료등급`(A/B/C) 분포를 아래 셀에서 확인했다."
))

cells.append(nbf.v4.new_code_cell(
    "final_df['원료등급'].value_counts(normalize=True) * 100"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/01_eda_and_validation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/01_eda_and_validation.ipynb")
```

- [ ] **Step 2: 스크립트 실행해 노트북 생성**

Run: `python scripts/build_notebook.py`
Expected: `노트북 생성 완료: notebooks/01_eda_and_validation.ipynb` 출력, 파일 생성됨

- [ ] **Step 3: 노트북이 에러 없이 전체 실행되는지 검증 (자동화된 테스트)**

Run:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda_and_validation.ipynb --output executed_check.ipynb
```
Expected: 종료 코드 0, 에러 없음

- [ ] **Step 4: 검증용 임시 산출물 삭제**

Run: `rm notebooks/executed_check.ipynb`

- [ ] **Step 5: 커밋**

```bash
git add scripts/build_notebook.py notebooks/01_eda_and_validation.ipynb
git commit -m "docs: add EDA/validation notebook with ablation results and interview answers"
```

---

## 완료 기준

- [ ] `pytest -v` 전체 통과 (약 27개 테스트)
- [ ] `python -m src.pipeline` 실행 시 `data/processed/cleaned_features.csv` 생성, 정상범위 위반 0건
- [ ] `notebooks/01_eda_and_validation.ipynb`가 nbconvert로 에러 없이 실행됨
- [ ] `reports/figures/`에 결측 전/후, 분포, 상관관계 히트맵 PNG 생성 확인
- [ ] 설계 문서 §9 심층 인터뷰 3문항에 대한 답변이 노트북에 근거와 함께 정리되어 있음
